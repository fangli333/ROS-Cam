#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import numpy as np
import random
import os, sys
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, l2_loss, lpips_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from torch.utils.data import DataLoader
from utils.timer import Timer
from utils.loader_utils import FineSampler, get_stamp_list
import lpips
from utils.scene_utils import render_training_image
from time import time
import copy

#ROS-Cam
import torch.nn.functional as F
import math
import cv2
from utils.graphics_utils import getWorld2View2_GPU, getProjectionMatrix_GPU, quater2rotation
import open3d as o3d
from PIL import Image, ImageDraw
import lpips
lpips_fn = lpips.LPIPS(net='vgg').to('cuda')


from scene.dataset_readers import fetchPly, set_global_args


to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
def scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations, 
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, stage, tb_writer, train_iter,timer):
    first_iter = 0
    
    pcd_cube = fetchPly(f'{args.pt3dplyfolder}')
    
    scene.gaussians.create_from_pcd(pcd_cube, scene.cameras_extent)
    
    
    gaussians.training_setup(opt, scene.cameras_extent)
    if checkpoint:
        # breakpoint()
        if stage == "coarse" and stage not in checkpoint:
            print("start from fine stage, skip coarse stage.")
            # process is in the coarse stage, but start from fine stage
            return
        if stage in checkpoint: 
            (model_params, first_iter) = torch.load(checkpoint)
            gaussians.restore(model_params, opt)


    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    # viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0

    final_iter = train_iter
    
    progress_bar = tqdm(range(first_iter, final_iter), desc="Training progress")
    first_iter += 1

    viewpoint_stack_pop = []
    viewpoint_stack_all = scene.getTrainCameras() 
    train_cams = []
    test_cams = []
    for i in range(len(viewpoint_stack_all)):
        if i % 2 == 0:
            train_cams.append(viewpoint_stack_all[i])
        else:
            test_cams.append(viewpoint_stack_all[i])

    batch_size = opt.batch_size

    count = 0
    for iteration in range(first_iter, final_iter+1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    count +=1
                    viewpoint_index = (count ) % len(test_cams)
                    if (count //(len(test_cams))) % 2 == 0:
                        viewpoint_index = viewpoint_index
                    else:
                        viewpoint_index = len(test_cams) - viewpoint_index - 1
                    viewpoint = test_cams[viewpoint_index]
                    custom_cam.time = viewpoint.time
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer, stage=stage, cam_type=scene.dataset_type)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive) :
                    break
            except Exception as e:
                print(e)
                network_gui.conn = None

        iter_start.record()
        # gaussians.update_learning_rate(iteration)
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        idx = 0
        viewpoint_cams = []
        while idx < batch_size :    
            if not viewpoint_stack_pop:
                viewpoint_stack_pop = train_cams.copy()      
            viewpoint_cam = viewpoint_stack_pop.pop(randint(0,len(viewpoint_stack_pop)-1))
            viewpoint_cams.append(viewpoint_cam)
            idx +=1

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        images = []
        gt_images = []
        radii_list = []
        visibility_filter_list = []
        viewspace_point_tensor_list = []
        for viewpoint_cam in viewpoint_cams:
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, stage=stage,cam_type=scene.dataset_type)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            images.append(image.unsqueeze(0))
            if scene.dataset_type!="PanopticSports":
                gt_image = viewpoint_cam.original_image.cuda()
            else:
                gt_image  = viewpoint_cam['image'].cuda()
            
            gt_images.append(gt_image.unsqueeze(0))
            radii_list.append(radii.unsqueeze(0))
            visibility_filter_list.append(visibility_filter.unsqueeze(0))
            viewspace_point_tensor_list.append(viewspace_point_tensor)
        

        radii = torch.cat(radii_list,0).max(dim=0).values
        visibility_filter = torch.cat(visibility_filter_list).any(dim=0)
        image_tensor = torch.cat(images,0)
        gt_image_tensor = torch.cat(gt_images,0)
        # Loss
        # breakpoint()
        Ll1 = l1_loss(image_tensor, gt_image_tensor[:,:3,:,:])

        psnr_ = psnr(image_tensor, gt_image_tensor).mean().double()
        # norm
        

        loss = Ll1
        if stage == "fine" and hyper.time_smoothness_weight != 0:
            # tv_loss = 0
            tv_loss = gaussians.compute_regulation(hyper.time_smoothness_weight, hyper.l1_time_planes, hyper.plane_tv_weight)
            loss += tv_loss
        if opt.lambda_dssim != 0:
            ssim_loss = ssim(image_tensor,gt_image_tensor)
            loss += opt.lambda_dssim * (1.0-ssim_loss)
        # if opt.lambda_lpips !=0:
        #     lpipsloss = lpips_loss(image_tensor,gt_image_tensor,lpips_model)
        #     loss += opt.lambda_lpips * lpipsloss
        
        loss.backward()
        if torch.isnan(loss).any():
            print("loss is nan,end training, reexecv program now.")
            os.execv(sys.executable, [sys.executable] + sys.argv)
        viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
        for idx in range(0, len(viewspace_point_tensor_list)):
            viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_psnr_for_log = 0.4 * psnr_ + 0.6 * ema_psnr_for_log
            total_point = gaussians._xyz.shape[0]
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}",
                                          "psnr": f"{psnr_:.{2}f}",
                                          "point":f"{total_point}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            timer.pause()
            #ROS-Cam
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, [pipe, background], stage, scene.dataset_type, train_cams, test_cams)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, stage)
            if dataset.render_process:
                if (iteration < 1000 and iteration % 10 == 9) \
                    or (iteration < 3000 and iteration % 50 == 49) \
                        or (iteration < 60000 and iteration %  100 == 99) :
                    # breakpoint()
                        render_training_image(scene, gaussians, [test_cams[iteration%len(test_cams)]], render, pipe, background, stage+"test", iteration,timer.get_elapsed_time(),scene.dataset_type)
                        render_training_image(scene, gaussians, [train_cams[iteration%len(train_cams)]], render, pipe, background, stage+"train", iteration,timer.get_elapsed_time(),scene.dataset_type)
                        # render_training_image(scene, gaussians, train_cams, render, pipe, background, stage+"train", iteration,timer.get_elapsed_time(),scene.dataset_type)

                    # total_images.append(to8b(temp_image).transpose(1,2,0))
            timer.start()
            # Densification
            if iteration < opt.densify_until_iter :
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)

                if stage == "coarse":
                    opacity_threshold = opt.opacity_threshold_coarse
                    densify_threshold = opt.densify_grad_threshold_coarse
                else:    
                    opacity_threshold = opt.opacity_threshold_fine_init - iteration*(opt.opacity_threshold_fine_init - opt.opacity_threshold_fine_after)/(opt.densify_until_iter)  
                    densify_threshold = opt.densify_grad_threshold_fine_init - iteration*(opt.densify_grad_threshold_fine_init - opt.densify_grad_threshold_after)/(opt.densify_until_iter )  
                if  iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 and gaussians.get_xyz.shape[0]<360000:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    
                    gaussians.densify(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold, 5, 5, scene.model_path, iteration, stage)
                if  iteration > opt.pruning_from_iter and iteration % opt.pruning_interval == 0 and gaussians.get_xyz.shape[0]>200000:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None

                    gaussians.prune(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)
                    
                # if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 :
                if iteration % opt.densification_interval == 0 and gaussians.get_xyz.shape[0]<360000 and opt.add_point:
                    gaussians.grow(5,5,scene.model_path,iteration,stage)
                    # torch.cuda.empty_cache()
                if iteration % opt.opacity_reset_interval == 0:
                    print("reset opacity")
                    gaussians.reset_opacity()
                    
            

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                gaussians.update_learning_rate(iteration)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" +f"_{stage}_" + str(iteration) + ".pth")


def camera_learn(dataset, opt, hyper, pipe, gaussians, scene, timer):
    gaussians.training_setup(opt, torch.tensor(5, device='cuda'))

    viewpoint_stack = scene.getTrainCameras()
    for item in viewpoint_stack:
        item.training_setup(opt)

    total_frame = len(viewpoint_stack)
        
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    progress_bar = tqdm(range(opt.camera_learn), desc="Camera Learning")


    for iteration in range(1, opt.camera_learn+1):      
        loss = 0
        iter_start.record()

        cali_points = gaussians.get_xyz  
        
        pad_cali_points = F.pad(cali_points, (0, 1), value=1)
        all_world = torch.tensor([], dtype = torch.float32).cuda()
        all_gt_pt = torch.tensor([], dtype = torch.float32).cuda()
        all_full = torch.empty(total_frame, 4, 4).cuda()
        all_r_trans = torch.tensor([]).cuda()
        all_camera_centers = torch.tensor([]).cuda()
        all_index = torch.tensor([]).cuda()
        
        p_matrix = getProjectionMatrix_GPU(znear=viewpoint_stack[0].znear, zfar=viewpoint_stack[0].zfar, fovX=2 * torch.atan(viewpoint_stack[0].image_width / (2 * viewpoint_stack[0].focal)), fovY=2 * torch.atan(viewpoint_stack[0].image_height / (2 * viewpoint_stack[0].focal)), device = viewpoint_stack[0].data_device).transpose(0, 1)

        for i in range(total_frame):
            all_r_trans = torch.cat((all_r_trans, quater2rotation(viewpoint_stack[i].quaternion).transpose(0, 1).unsqueeze(0)), dim = 0) 
            wwtransform = getWorld2View2_GPU(R = all_r_trans[i], t = viewpoint_stack[i].T, translate = viewpoint_stack[i].trans, scale = viewpoint_stack[i].scale, device = viewpoint_stack[i].data_device).transpose(0,1)
            all_full[i] = (wwtransform.unsqueeze(0).bmm(p_matrix.unsqueeze(0))).squeeze(0)
            all_world = torch.cat((all_world, pad_cali_points[viewpoint_stack[i].index].unsqueeze(0)), dim = 0)
            all_gt_pt = torch.cat((all_gt_pt, viewpoint_stack[i].pt.unsqueeze(0)), dim = 0)   #h,w
            all_camera_centers = torch.cat((all_camera_centers, wwtransform.inverse()[3, :3].unsqueeze(0)), dim = 0)
            all_index = torch.cat((all_index, viewpoint_stack[i].index.unsqueeze(0)), dim = 0) 

        
        #projection
        all_project_p = torch.matmul(all_world, all_full)
        all_image_p = all_project_p[:, :, :2] / all_project_p[:, :, 3].unsqueeze(2)  #w, h
        al_hw_p = all_image_p[:, :, [1, 0]]  #h,w
        all_hw_loca = al_hw_p * torch.tensor([viewpoint_stack[0].image_height / 2, viewpoint_stack[0].image_width / 2]).cuda() + torch.tensor([viewpoint_stack[0].image_height / 2, viewpoint_stack[0].image_width / 2]).cuda()   #cuda h,w

        #projection l2
        proj_loss = torch.sum((all_hw_loca - all_gt_pt) ** 2, dim=2, keepdim=True).squeeze(2) 

        #depth regularization
        depth_reg = torch.sum(torch.relu(- all_project_p[:, :, 3]))

        loss_cali = torch.zeros(pad_cali_points.shape[0], device='cuda')
        loss_cali.index_add_(0, all_index.view(-1).long(), proj_loss.view(-1))  #torch.Size([1721])
        loss_cali = loss_cali / all_index.view(-1).long().bincount(minlength=loss_cali.size(0))


        #Gamma_raw
        if iteration <= opt.coarse_camera_learn:  
            raw_gamma = torch.zeros(cali_points.shape[0], device='cuda')

        if iteration == opt.coarse_camera_learn + 1:
            gaussians.reset_gamma(loss_cali)
        
        if iteration > opt.coarse_camera_learn:
            raw_gamma = gaussians.get_raw_gamma

        gamma = F.softplus(raw_gamma)

        cauchy_loss_cali = torch.log(gamma + loss_cali ** 2 / gamma).mean()
        loss += (cauchy_loss_cali + 10.0 * depth_reg)


        # print('--------Projection LOSS--------')
        # print(proj_loss.mean())
        # print(depth_reg)
        # print('----------Cali loss----------')
        # print(loss_cali.mean())
        # print(loss_cali.min(), loss_cali.max())
        # print('--------Cauchy LOSS--------')
        # print(cauchy_loss_cali)

        if iteration == opt.camera_learn:
            avg_cam_center = torch.mean(all_camera_centers, dim=0, keepdim=True)
            cam_dist = torch.norm(all_camera_centers - avg_cam_center, p=2, dim = 1)
            cam_diagonal = torch.max(cam_dist)
            cam_radius = cam_diagonal * 1.1
            scene.cameras_extent = cam_radius

        loss.backward() 
        iter_end.record()
        
        # Update progress bar
        with torch.no_grad():
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{loss.item():.7f}"})
                progress_bar.update(10)
            if iteration == opt.camera_learn:
                progress_bar.close()
        
        if opt.coarse_camera_learn < iteration <= opt.camera_learn and iteration % 50 == 0:
            all_image_loca = all_hw_loca.detach().to(dtype=torch.long)
            all_image_loca[:, :, 0] = torch.clamp(all_image_loca[:, :, 0], min = 0, max = viewpoint_stack[0].image_height - 1)
            all_image_loca[:, :, 1] = torch.clamp(all_image_loca[:, :, 1], min = 0, max = viewpoint_stack[0].image_width - 1)
            all_image_loca = all_image_loca.to(torch.int32).cpu().detach().numpy()
            all_gt_loca = all_gt_pt.detach().to(torch.int32).cpu().numpy()

            assert all_image_loca.shape[0] == total_frame

            base_dir = f'{args.camerafolder}'
            subdirs = ['2dprojection', 'R', 'T', 'CameraCenter', 'Focal', 'otherpoint', 'gamma']

            for sub in subdirs:
                os.makedirs(os.path.join(base_dir, sub), exist_ok=True)

            for j in range(total_frame):
                image = (viewpoint_stack[j].original_image * 255).clamp(0, 255).permute(1, 2, 0)
                image = Image.fromarray(image.cpu().detach().numpy().astype(np.uint8))
                draw = ImageDraw.Draw(image)
                x_size = 5

                gamma_current_frame = gamma[viewpoint_stack[j].index].detach().cpu().numpy()

                for w in range(all_image_loca[j].shape[0]):
                    x, y = all_image_loca[j][w]
                    x_gt, y_gt = all_gt_loca[j][w]
                    gamma_current = np.round(gamma_current_frame[w], 2)

                    draw.line((y - x_size, x - x_size, y + x_size, x + x_size), fill="green", width=2)
                    draw.line((y - x_size, x + x_size, y + x_size, x - x_size), fill="green", width=2)
                    draw.text((y, x), str(gamma_current), fill="green")
                    draw.line((y_gt - x_size, x_gt - x_size, y_gt + x_size, x_gt + x_size), fill="blue", width=2)
                    draw.line((y_gt - x_size, x_gt + x_size, y_gt + x_size, x_gt - x_size), fill="blue", width=2)

                # Save the current frame's image and camera parameters
                image.save(f'{base_dir}/2dprojection/{j:04d}.png')
                viewpoint_stack[j].wp_compute()
                np.save(f'{base_dir}/R/{j:04d}.npy', viewpoint_stack[j].R.detach().cpu().numpy())  # from camera to world
                np.save(f'{base_dir}/T/{j:04d}.npy', viewpoint_stack[j].T.detach().cpu().numpy())  # from world to camera
                np.save(f'{base_dir}/CameraCenter/{j:04d}.npy', viewpoint_stack[j].camera_center.detach().cpu().numpy())  # camera location in world coordinate

            # shared focal and other data
            np.save(f'{base_dir}/Focal/focal.npy', viewpoint_stack[0].focal.detach().cpu().numpy())
            np.save(f'{base_dir}/otherpoint/cali_points.npy', cali_points.detach().cpu().numpy())
            np.save(f'{base_dir}/gamma/gamma.npy', gamma.detach().cpu().numpy())

            print('Saving finished!')
        
            
        with torch.no_grad():
            if iteration <= opt.camera_learn:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                gaussians.update_learning_rate_cali(iteration)
                for i in range(total_frame):
                    viewpoint_stack[i].optimizer.step()
                    viewpoint_stack[i].optimizer.zero_grad()
                    viewpoint_stack[i].update_learning_rate(iteration)





def training(dataset, hyper, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, expname):
    # first_iter = 0
    tb_writer = prepare_output_and_logger(expname)
    gaussians = GaussianModel(dataset.sh_degree, hyper)
    dataset.model_path = args.model_path
    timer = Timer()
    scene = Scene(dataset, gaussians, load_coarse=None)
    timer.start()


    camera_learn(dataset, opt, hyper, pipe, gaussians, scene, timer)

    scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                             checkpoint_iterations, checkpoint, debug_from,
                             gaussians, scene, "coarse", tb_writer, opt.coarse_iterations,timer)
    scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, "fine", tb_writer, opt.iterations,timer)

def prepare_output_and_logger(expname):    
    if not args.model_path:
        # if os.getenv('OAR_JOB_ID'):
        #     unique_str=os.getenv('OAR_JOB_ID')
        # else:
        #     unique_str = str(uuid.uuid4())
        unique_str = expname

        args.model_path = os.path.join("./output/", unique_str)
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, stage, dataset_type, train_cams, test_cams):
    if tb_writer:
        tb_writer.add_scalar(f'{stage}/train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'{stage}/train_loss_patchestotal_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'{stage}/iter_time', elapsed, iteration)
        
    
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()

        validation_configs = ({'name': 'test', 'cameras' : test_cams},
                              {'name': 'train', 'cameras' : train_cams})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians,stage=stage, cam_type=dataset_type, *renderArgs)["render"], 0.0, 1.0)
                    if dataset_type == "PanopticSports":
                        gt_image = torch.clamp(viewpoint["image"].to("cuda"), 0.0, 1.0)
                    else:
                        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    try:
                        if tb_writer and (idx < 5):
                            tb_writer.add_images(stage + "/"+config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                            if iteration == testing_iterations[0]:
                                tb_writer.add_images(stage + "/"+config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    except:
                        pass
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image, mask=None).mean().double()

                    ssim_test += ssim(image, gt_image).mean().double()
                    lpips_test += lpips_fn(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])   

                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])


                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {}".format(iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))


                # print("sh feature",scene.gaussians.get_features.shape)
                if tb_writer:
                    tb_writer.add_scalar(stage + "/"+config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(stage+"/"+config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram(f"{stage}/scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            
            tb_writer.add_scalar(f'{stage}/total_points', scene.gaussians.get_xyz.shape[0], iteration)
            tb_writer.add_scalar(f'{stage}/deformation_rate', scene.gaussians._deformation_table.sum()/scene.gaussians.get_xyz.shape[0], iteration)
            tb_writer.add_histogram(f"{stage}/scene/motion_histogram", scene.gaussians._deformation_accum.mean(dim=-1)/100, iteration,max_bins=500)
        
        torch.cuda.empty_cache()
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
if __name__ == "__main__":
    # Set up command line argument parser
    # torch.set_default_tensor_type('torch.FloatTensor')
    torch.cuda.empty_cache()
    parser = ArgumentParser(description="Training script parameters")
    setup_seed(6666)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[3000,7000,14000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[ 14000, 20000, 30_000, 45000, 60000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--expname", type=str, default = "")
    parser.add_argument("--configs", type=str, default = "")
    ###ROS-Cam
    parser.add_argument("--ptidxfolder", type=str, default = "")
    parser.add_argument("--pt3dplyfolder", type=str, default = "")
    parser.add_argument("--camerafolder", type=str, default = "")
    
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Set global args for dataset readers (ROS-Cam)
    set_global_args(args)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.expname)

    # All done
    print("\nTraining complete.")

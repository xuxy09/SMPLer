""" Author: Xiangyu Xu
    For rendering meshes with PyTorch3D
"""

from pytorch3d.renderer.mesh import textures
import torch
import torch.nn as nn
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (OrthographicCameras, TexturesVertex, DirectionalLights, 
                                RasterizationSettings, MeshRenderer, PerspectiveCameras,
                                MeshRasterizer, SoftPhongShader, HardPhongShader)
import constants


color_list = [(0.63, 0.93, 0.88), (0.93, 0.68, 0.62)]

class Pt3dRenderer:
    def __init__(self, faces, device, img_size):
        self.device = device
        self.faces = faces.to(device)  # [1, num_faces, 3]
        self.img_size = img_size

        # for perspective
        self.R = torch.eye(3, device=self.device)
        self.R[0, 0] = -1
        self.R[1, 1] = -1
        self.R = self.R.view(1, 3, 3)
        self.focal_length = torch.tensor([constants.focal_length / (constants.img_res / 2)], device=self.device)
        self.principal_point = torch.tensor([0, 0], device=self.device).view(1, 2)

        # prepare renderer
        lights = DirectionalLights(direction=((0,0,-1),), ambient_color=((0.3,0.3,0.3),), diffuse_color=((0.6,0.6,0.6),), specular_color=((0.1,0.1,0.1),)).to(self.device)
        raster_settings = RasterizationSettings(image_size=img_size, blur_radius=0.0, faces_per_pixel=1)
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(raster_settings=raster_settings),
            shader=HardPhongShader(device=device, lights=lights)
        )

        self.render = self.render_orthographic

    def render_orthographic(self, verts, cam, color=(0.63, 0.93, 0.88), img_size=None):
        # verts: [B, num_verts, 3]
        # cam: [B, 3]

        if img_size is None:
            raster_settings = RasterizationSettings(image_size=self.img_size, blur_radius=0.0, faces_per_pixel=1)
        else:
            raster_settings = RasterizationSettings(image_size=img_size, blur_radius=0.0, faces_per_pixel=1)

        B = verts.shape[0]
        num_verts = verts.shape[1]
        verts_ = verts.clone()
        verts_[..., 2] = verts[..., 2] + 10  # avoid clipping when z < 0; as we use weak perspective projection, z has no gradients

        vertex_color = torch.tensor(color).view(1,1,-1).repeat(B, num_verts, 1) 
        textures = TexturesVertex(vertex_color).to(self.device)
        meshes = Meshes(verts_, self.faces.expand(B, -1, -1), textures=textures)
        
        # update camera: cam to pytorch3d-toNDC-projection-matrix
        K = torch.zeros(B, 4, 4, device=self.device)
        K[:, 0, 0] = -cam[:, 0] 
        K[:, 0, 3] = -cam[:, 0] * cam[:, 1] 
        K[:, 1, 1] = -cam[:, 0] 
        K[:, 1, 3] = -cam[:, 0] * cam[:, 2] 
        K[:, 2, 2] = 1
        K[:, 3, 3] = 1
        cameras = OrthographicCameras(K=K, in_ndc=True, device=self.device)
        
        rendered_imgs = self.renderer(meshes, cameras=cameras, raster_settings=raster_settings)  # [B, img_size, img_size, 4:RGBA]
        return rendered_imgs






        

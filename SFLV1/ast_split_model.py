import timm
import torch
import torch.nn as nn
# from torch.utils.data import DataLoader, random_split
# import torch.optim as optim
from torch.amp import autocast
# from tqdm import tqdm
from timm.models.layers import to_2tuple,trunc_normal_

label_dim = 2
input_tdim = 259
input_fdim = 128
#-----------------------------------------------------------------------------------------------------------------------------#
class PatchEmbed(nn.Module):
    def __init__(self, img_size=(128, 259), patch_size=16, in_chans=1, embed_dim=768):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
#-----------------------------------------------------------------------------------------------------------------------------#
class ASTClient(nn.Module):
    def __init__(self, fstride=10, tstride=10, input_fdim=128, input_tdim=1024, imagenet_pretrain=True, model_size='tiny224', verbose=False):
        super(ASTClient, self).__init__()
        timm.models.vision_transformer.PatchEmbed = PatchEmbed
        if model_size == 'tiny224':
            self.v = timm.create_model('vit_deit_tiny_distilled_patch16_224', pretrained=imagenet_pretrain)
        elif model_size == 'small224':
            self.v = timm.create_model('vit_deit_small_distilled_patch16_224', pretrained=imagenet_pretrain)
        elif model_size == 'base224':
            self.v = timm.create_model('vit_deit_base_distilled_patch16_224', pretrained=imagenet_pretrain)
        elif model_size == 'base384':
            self.v = timm.create_model('vit_deit_base_distilled_patch16_384', pretrained=imagenet_pretrain)
        else:
            raise Exception('Model size must be one of tiny224, small224, base224, base384.')
        self.original_num_patches = self.v.patch_embed.num_patches
        self.oringal_hw = int(self.original_num_patches ** 0.5)
        self.original_embedding_dim = self.v.pos_embed.shape[2]
        self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, label_dim))

        

        # automatcially get the intermediate shape
        f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
        num_patches = f_dim * t_dim
        self.v.patch_embed.num_patches = num_patches
        if verbose == True:
            print('frequncey stride={:d}, time stride={:d}'.format(fstride, tstride))
            print('number of patches={:d}'.format(num_patches))

        # the linear projection layer
        new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        if imagenet_pretrain == True:
            new_proj.weight = torch.nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
            new_proj.bias = self.v.patch_embed.proj.bias
        self.v.patch_embed.proj = new_proj

        # the positional embedding
        if imagenet_pretrain == True:
            # get the positional embedding from deit model, skip the first two tokens (cls token and distillation token), reshape it to original 2D shape (24*24).
            new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, self.original_num_patches, self.original_embedding_dim).transpose(1, 2).reshape(1, self.original_embedding_dim, self.oringal_hw, self.oringal_hw)
            # cut (from middle) or interpolate the second dimension of the positional embedding
            if t_dim <= self.oringal_hw:
                new_pos_embed = new_pos_embed[:, :, :, int(self.oringal_hw / 2) - int(t_dim / 2): int(self.oringal_hw / 2) - int(t_dim / 2) + t_dim]
            else:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(self.oringal_hw, t_dim), mode='bilinear')
            # cut (from middle) or interpolate the first dimension of the positional embedding
            if f_dim <= self.oringal_hw:
                new_pos_embed = new_pos_embed[:, :, int(self.oringal_hw / 2) - int(f_dim / 2): int(self.oringal_hw / 2) - int(f_dim / 2) + f_dim, :]
            else:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
            # flatten the positional embedding
            new_pos_embed = new_pos_embed.reshape(1, self.original_embedding_dim, num_patches).transpose(1,2)
            # concatenate the above positional embedding with the cls token and distillation token of the deit model.
            self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))            
        else:
            # if not use imagenet pretrained model, just randomly initialize a learnable positional embedding
            # TODO can use sinusoidal positional embedding instead
            new_pos_embed = nn.Parameter(torch.zeros(1, self.v.patch_embed.num_patches + 2, self.original_embedding_dim))
            self.v.pos_embed = new_pos_embed
            trunc_normal_(self.v.pos_embed, std=.02)

    def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=1024):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

    @autocast('cuda')
    def forward(self, x):
        # x = x.unsqueeze(1)
        x = x.transpose(2, 3)
        x = self.v.patch_embed(x)
        B = x.shape[0]
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)
        
        

        return x
        
#-----------------------------------------------------------------------------------------------------------------------------#
class ASTServer(nn.Module):
    def __init__(self, fstride=10, tstride=10, input_fdim=128, input_tdim=1024, imagenet_pretrain=True, model_size='tiny224', verbose=False):
        super(ASTServer, self).__init__()
        timm.models.vision_transformer.PatchEmbed = PatchEmbed
        if model_size == 'tiny224':
            self.v = timm.create_model('vit_deit_tiny_distilled_patch16_224', pretrained=imagenet_pretrain)
        elif model_size == 'small224':
            self.v = timm.create_model('vit_deit_small_distilled_patch16_224', pretrained=imagenet_pretrain)
        elif model_size == 'base224':
            self.v = timm.create_model('vit_deit_base_distilled_patch16_224', pretrained=imagenet_pretrain)
        elif model_size == 'base384':
            self.v = timm.create_model('vit_deit_base_distilled_patch16_384', pretrained=imagenet_pretrain)
        else:
            raise Exception('Model size must be one of tiny224, small224, base224, base384.')
        self.original_num_patches = self.v.patch_embed.num_patches
        self.oringal_hw = int(self.original_num_patches ** 0.5)
        self.original_embedding_dim = self.v.pos_embed.shape[2]
        self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, label_dim))

        # automatcially get the intermediate shape
        f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
        print("f_dim = {}, t_dim = {}".format(f_dim, t_dim))
        num_patches = f_dim * t_dim
        self.v.patch_embed.num_patches = num_patches
        if verbose == True:
            print('frequncey stride={:d}, time stride={:d}'.format(fstride, tstride))
            print('number of patches={:d}'.format(num_patches))

        # the linear projection layer
        new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        if imagenet_pretrain == True:
            new_proj.weight = torch.nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
            new_proj.bias = self.v.patch_embed.proj.bias
        self.v.patch_embed.proj = new_proj

        # the positional embedding
        if imagenet_pretrain == True:
            # get the positional embedding from deit model, skip the first two tokens (cls token and distillation token), reshape it to original 2D shape (24*24).
            new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, self.original_num_patches, self.original_embedding_dim).transpose(1, 2).reshape(1, self.original_embedding_dim, self.oringal_hw, self.oringal_hw)
            # cut (from middle) or interpolate the second dimension of the positional embedding
            if t_dim <= self.oringal_hw:
                new_pos_embed = new_pos_embed[:, :, :, int(self.oringal_hw / 2) - int(t_dim / 2): int(self.oringal_hw / 2) - int(t_dim / 2) + t_dim]
            else:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(self.oringal_hw, t_dim), mode='bilinear')
            # cut (from middle) or interpolate the first dimension of the positional embedding
            if f_dim <= self.oringal_hw:
                new_pos_embed = new_pos_embed[:, :, int(self.oringal_hw / 2) - int(f_dim / 2): int(self.oringal_hw / 2) - int(f_dim / 2) + f_dim, :]
            else:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
            # flatten the positional embedding
            new_pos_embed = new_pos_embed.reshape(1, self.original_embedding_dim, num_patches).transpose(1,2)
            # concatenate the above positional embedding with the cls token and distillation token of the deit model.
            self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))            
        else:
            # if not use imagenet pretrained model, just randomly initialize a learnable positional embedding
            # TODO can use sinusoidal positional embedding instead
            new_pos_embed = nn.Parameter(torch.zeros(1, self.v.patch_embed.num_patches + 2, self.original_embedding_dim))
            self.v.pos_embed = new_pos_embed
            trunc_normal_(self.v.pos_embed, std=.02)

    def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=1024):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim
    
    @autocast('cuda')
    def forward(self, x):
        
        x = (x[:, 0] + x[:, 1]) / 2

        x = self.mlp_head(x)
        return x
#-----------------------------------------------------------------------------------------------------------------------------#
class ASTFullModel(nn.Module):
    def __init__(self, input_fdim=128, input_tdim=259, fstride=10, tstride=10, imagenet_pretrain=True, model_size='tiny224'):
        super(ASTFullModel, self).__init__()
        self.client = ASTClient(fstride=fstride, tstride=tstride, input_fdim=input_fdim, input_tdim=input_tdim,
                                imagenet_pretrain=imagenet_pretrain, model_size=model_size)
        self.server = ASTServer(fstride=fstride, tstride=tstride, input_fdim=input_fdim, input_tdim=input_tdim,
                                imagenet_pretrain=imagenet_pretrain, model_size=model_size)

    def forward(self, x):
        x = self.client(x)
        x = self.server(x)
        return x
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# if __name__ == "__main__":
#     # 数据集加载
#     dataset = astDataWrap.dat_wrap(astDataWrap.fetch_ast_mel_dat())
#     train_size = int(0.64 * len(dataset))
#     val_size = int(0.16 * len(dataset))
#     test_size = len(dataset) - train_size - val_size
#     train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size)

#     # 模型初始化
#     client = ASTClient(input_fdim=input_fdim, input_tdim=input_tdim).to(device)
#     server = ASTServer(input_fdim=input_fdim, input_tdim=input_tdim).to(device)

#     # Loss & Optimizer
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(list(client.parameters()) + list(server.parameters()), lr=1e-4)

#     # 开始训练
#     for epoch in range(num_epochs):
#         client.train()
#         server.train()
#         total_loss = 0
#         correct_train = 0
#         total_train = 0

#         for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
#             x, y = x.to(device), y.to(device)  # [B, 128, 259] -> [B, 1, 128, 259]
#             optimizer.zero_grad()

#             client_out = client(x)
#             output = server(client_out)

#             loss = criterion(output, y)
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()
#             pred = output.argmax(dim=1)
#             correct_train += (pred == y).sum().item()
#             total_train += y.size(0)

#         train_acc = 100 * correct_train / total_train
#         print(f'Epoch {epoch+1} Train Loss: {total_loss / len(train_loader):.4f} | Train Acc: {train_acc:.2f}%')

#         # 验证
#         client.eval()
#         server.eval()
#         correct_val = 0
#         total_val = 0
#         with torch.no_grad():
#             for x, y in val_loader:
#                 x, y = x.to(device).float(), y.to(device)
#                 client_out = client(x)
#                 output = server(client_out)
#                 pred = output.argmax(dim=1)
#                 correct_val += (pred == y).sum().item()
#                 total_val += y.size(0)
#         val_acc = 100 * correct_val / total_val
#         print(f'Epoch {epoch+1} Valid Accuracy: {val_acc:.2f}%')

#         # 测试
#         correct_test = 0
#         total_test = 0
#         with torch.no_grad():
#             for x, y in test_loader:
#                 x, y = x.to(device).float(), y.to(device)
#                 client_out = client(x)
#                 output = server(client_out)
#                 pred = output.argmax(dim=1)
#                 correct_test += (pred == y).sum().item()
#                 total_test += y.size(0)
#         test_acc = 100 * correct_test / total_test
#         print(f'Epoch {epoch+1} Test Accuracy: {test_acc:.2f}%')
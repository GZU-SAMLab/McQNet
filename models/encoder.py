

def setup_backbone(name, pretrained=True):
    if name == "resnet18":
        model = torchvision.models.resnet18(pretrained=False)
        model.load_state_dict(torch.load('/groups/public_cluster/home/samuel/projects/MyNet/pretrained_weights/resnet18-5c106cde.pth'))
        model.fc = Flatten()
        return model
    elif name == "ViT":
        model = create_model(num_classes=1000, has_logits=False)
        model.load_state_dict(
            torch.load('/groups/public_cluster/home/samuel/projects/MyNet/pretrained_weights/jx_vit_base_p16_224-80ecf9dd.pth'))
        return model
    elif name == "swin":
        model = swin(num_classes=1000, has_logits=False)
        model.load_state_dict(
            torch.load( '/groups/public_cluster/home/samuel/projects/MyNet/pretrained_weights/swin_base_patch4_window7_224.pth')["model"], strict=False)
        return model
    else:
        raise NotImplementedError("this option is not defined")
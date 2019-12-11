from utils import *
from model import *


# desired depth layers to compute style/content losses :
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)

    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]
    print(model)

    return model, style_losses, content_losses


def get_style_model_and_losses_gan(gan, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    gan = copy.deepcopy(gan)

    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    #model = nn.Sequential(normalization)
    model = nn.Sequential()
#     #   (0): ReflectionPad2d((3, 3, 3, 3))
#     (1): Conv2d(3, 64, kernel_size=(7, 7), stride=(1, 1))
#     (2): InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
#     (3): ReLU(inplace=True)
#     (4): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
#     (5): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
#     (6): ReLU(inplace=True)
#     (7): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    i = 0  # increment every time we see a conv
    for layer in gan.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        elif isinstance(layer, nn.ReflectionPad2d):
            name = 'ref_{}'.format(i)
        elif isinstance(layer, nn.InstanceNorm2d):
            name = 'insNorm_{}'.format(i)
        elif isinstance(layer, ResidualBlock):
            name = "residual_{}".format(i)
        else:
            print("stop loading model")
            break
           #name = 'res_{}'.format(i)
            #raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
        #print(name)

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]
    print(model)
    return model, style_losses, content_losses


''' gan model + fine tune layers '''
def get_style_model_and_losses_gan_finetue(ganModel, content_img, style_img):
    model = copy.deepcopy(ganModel)
    #model = nn.Sequential()
    content_losses = []
    style_losses = []

    child_counter = 0
    for child in ganModel.children():
        child_counter += 1
        if child_counter == 27:
            # model.add_module("tmp", child)
            print(child)
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_" + str(child_counter), content_loss)
            content_losses.append(content_loss)

            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_" + str(child_counter), style_loss)
            style_losses.append(style_loss)

    child_counter = 0
    for child in model.children():
        child_counter += 1
        if child_counter < 25:
            for param in child.parameters():
                param.requires_grad = False
        else:
            print(child)
            for param in child.parameters():
                param.requires_grad = True
    return model, style_losses, content_losses


''' run style transfer '''
def run_style_transfer(net, normalization_mean, normalization_std,
                       content_img, style_img, input_img, content_layers, style_layers, lossFile, model_type, optimizer, num_steps=300,
                       style_weight=1000000, content_weight=1):
    """Run the style transfer."""
    print('Building the style transfer model..')
    #model, style_losses, content_losses = get_style_model_and_losses(cnn,
     #   normalization_mean, normalization_std, style_img, content_img)
    if model_type == "cnn":
        model, style_losses, content_losses = get_style_model_and_losses(net, normalization_mean, normalization_std,
                                                                         style_img, content_img,
                                                                         content_layers = content_layers,
                                                                         style_layers = style_layers)
    elif model_type.endswith("finetune"):
        model, style_losses, content_losses = get_style_model_and_losses_gan_finetue(net, content_img, style_img)
    else:
        model, style_losses, content_losses = get_style_model_and_losses_gan(net, normalization_mean, normalization_std,
                                                                         style_img, content_img,
                                                                         content_layers=content_layers,
                                                                         style_layers=style_layers)
    print(input_img)
    optimizer = get_input_optimizer(input_img, optimizer = optimizer)

    print('Optimizing..')
    print('iteration step : {0}'.format(num_steps))
    run = [0]
    f = open(lossFile, 'w')
    f.write("run\tstyleLoss\tcontentLoss\n")
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight
            if run[0] == 0:
                print("run {}:".format("[0]"))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()
                f.write("{}\t{:4f}\t{:4f}\n".format(run[0], style_score.item(), content_score.item()))

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            f.write("{}\t{:4f}\t{:4f}\n".format(run[0], style_score.item(), content_score.item()))
            #print(run[0])
            if run[0] % 10 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    f.close()
    # a last correction...
    input_img.data.clamp_(0, 1)
    print(input_img)

    return input_img


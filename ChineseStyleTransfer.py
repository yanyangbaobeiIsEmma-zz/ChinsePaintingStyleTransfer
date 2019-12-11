## CNN neural style transfer
from transfer import *
from utils import *
from model import  *
from torchvision.utils import save_image
from torch.autograd import Variable
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

''' model type '''
'''

1. cnn-neural (vgg-19) neural transfer (require style_img)
2. gongbi-gan
3. shuimo-gan
4. cnn-gongbi  (require style_img, naive combination)
5. cnn-shuimo  (require style_img, naive combination)
6. gongbi-neural (require style_img)
7. shuimo-neural (require style_img)
8. gongbi-gan-finetune
9. shuimo-gan-finetune

'''

if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", dest = "model_type", help = "Model type used for neural transfer", required = False, default = "cnn-neural")
    parser.add_argument("--content_img", dest = "content_img", help = "Content image for neural transfer", required = True)
    parser.add_argument("--style_img", dest = "style_img", help = "Style image for neural transfer", required = False)
    parser.add_argument("--output_folder", dest = "output_folder", help = "Output folder for loss file, generated painting", required = False, default = "./output/")
    parser.add_argument("--optimizer", dest = "optimizer", help = "Optimizer for neural transfer model", required = False, default = "SGD")
    parser.add_argument("--num_steps", dest = "num_steps", help = "Number of steps for iteration", required = False, default = 50)
    parser.add_argument("--style_weight", dest = "style_weight", help = "Weight for style loss", required = False, default = 1000000)
    parser.add_argument("--content_weight", dest = "content_weight", help = "Weight for content loss", required = False, default = 1)

    args = parser.parse_args()

    content_img = image_loader(args.content_img)
    content = args.content_img.split('/')[-1].split('.')[0]

    model_type = args.model_type
    optimizer = args.optimizer
    num_steps = int(args.num_steps)
    style_weight = int(args.style_weight)
    content_weight = int(args.content_weight)

    # mean and std for vgg-19
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    if model_type not in set(["cnn-neural", "gongbi-gan", "shuimo-gan", "cnn-gongbi", "cnn-shuimo", "gongbi-neural", "shuimo-neural", "gongbi-gan-finetune", "shuimo-gan-finetune"]):
        raise Exception("{} is not supported! Please choose from (cnn-neural, gongbi-gan, shuimo-gan, cnn-gongb, cnn-shuimo, gongbi-neural, shuimo-neural)".format(model_type))


    if model_type.endswith("gan"):
        output_path =  args.output_folder + '_'.join([content, model_type]).lstrip('_') + "_output.jpg"
        if "gongbi" in model_type:
            output, _ = gan_generator_eval("gongbi", args.content_img, output_path)
        else:
            output, _ = gan_generator_eval("shuimo", args.content_img, output_path)

    else:
        style = args.style_img.split('/')[-1].split('.')[0]
        style_img = image_loader(args.style_img)
        assert style_img.size() == content_img.size(), \
            "we need to import style and content images of the same size"

        lossFile = args.output_folder + '_'.join(
            ["content", content, "style", style, model_type, optimizer, str(args.num_steps), str(args.style_weight), str(args.content_weight)]) + "loss.tsv"

        if model_type.startswith("cnn"):
            ''' we use vgg-19 model '''
            content_layers = ['conv_4']
            style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
            optimizer = "LBFGS"
            transferModel = models.vgg19(pretrained=True).features.to(device).eval()
            if model_type == "cnn-neural": # cnn neural transfer
                input_img = content_img
            else: # naive combination
                tmp_output = args.output_folder + '_'.join(["content", content, "style", style, model_type, optimizer, str(args.num_steps), str(args.style_weight),
                                    str(args.content_weight)]) + "_intermediate_output.png"
                if "gongbi" in model_type:
                    input_img, _ = gan_generator_eval("gongbi", args.content_img, tmp_output)
                else:
                    input_img, _ = gan_generator_eval("shuimo", args.content_img, tmp_output)

        elif model_type.endswith("finetune"):
            ''' we use gan fine tune with style loss and content loss '''
            content_layers = []
            style_layers = []
            tmp_output = args.output_folder + '_'.join(
                ["content", content, "style", style, model_type, optimizer, str(args.num_steps), str(args.style_weight),
                 str(args.content_weight)]) + "intermediate_output.png"
            optimizer = "LBFGS"
            if "gongbi" in model_type:
                input_img, transferModel = gan_generator_eval("gongbi", args.content_img, tmp_output)
            else:
                input_img, transferModel = gan_generator_eval("shuimo", args.content_img, tmp_output)
            #transferModel = gan_finetue(ganModel, content_img, style_img)

        else:
            ''' we use gan generator model '''
            tmp_output = args.output_folder + '_'.join(["content", content, "style", style, model_type, optimizer, str(args.num_steps), str(args.style_weight),
                                   str(args.content_weight)]) + "intermediate_output.png"
            content_layers = ['relu_3']
            style_layers = ['conv_1', 'conv_2', 'conv_3']
            optimizer = "SGD"
            if "gongbi" in model_type:
                input_img, transferModel = gan_generator_eval("gongbi", args.content_img, tmp_output)
            else:
                input_img, transferModel = gan_generator_eval("shuimo", args.content_img, tmp_output)

        print("model_type= {}, style_weight = {}, content_weight = {}, num_steps = {}".format(model_type, style_weight, content_weight, num_steps))
        output = run_style_transfer(transferModel, cnn_normalization_mean, cnn_normalization_std,
                                    content_img, style_img, input_img, content_layers, style_layers, lossFile,
                                    model_type=model_type, optimizer=optimizer,
                                    num_steps=num_steps, style_weight=style_weight,
                                    content_weight=content_weight)
    plt.figure()
    imshow(output, title=model_type + ' Output Image')
    plt.ioff()
    plt.show()
    output_path = args.output_folder + '_'.join(["content", content, "style", style, model_type, optimizer, str(args.num_steps), str(args.style_weight),
                                   str(args.content_weight)]) + "_output.jpg"
    save_image(output, output_path)


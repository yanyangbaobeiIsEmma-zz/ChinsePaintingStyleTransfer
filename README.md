# ChinsePaintingStyleTransfer
This is CS 236 2019 Fall class project from Emma (Yanyang) Kong and Kris(Weijian) Ma.

Artistic style transfer aims to modify the style of the image while preserving its content. Style transfer using deep learning models has been widely studied since 2015, and most of the applications are focused on specific artists like Van Gogh, Monet, Cezanne. There are few researches and applications on traditional Chinese painting style transfer. In this paper, we will study and leverage different state-of-the-art deep generative models for Chinese painting style transfer and evaluate the performance both qualitatively and quantitatively. In addition, we propose our own algorithm that combines several style transfer models for our task. Specifically, we will transfer two main types of traditional Chinese painting style, known as "Gong-bi"(工笔) and "Shui-mo" (水墨) to modern images like nature objects, portraits and landscapes.

Traditional Chinese painting known as "Guo-hua"(国画) is very different from Western styles of art. There are two main styles in Chinese painting, one is "Gong-bi"(工笔), which means "meticulous". It uses detailed brushstrokes to precisely depict figures and subjects. The other one is "Shui-mo"(水墨), meaning "water and ink", which is close to freehand style sketch using brush.

In this project We start with preparing our dataset, which includes source dataset (realistic photos of people, flowers, birds, and landscapes), and target dataset (Chinese paintings including "Gong-bi"(工笔) and "Shui-mo"(水墨)). First we will examine the model introduced by Gatys et al. in both "Gong-bi" and "Shui-mo" paintings. Then we train cycle-GAN with our own dataset in order to convert a realistic photo to Chinese painting. Since "Gong-bi" and "Shui-mo" differ from each another in their colors, narrative subjects, level of details and so on, we train these two categories separately. We will compare the performance of Gatys et al. CNN model with cycle-GANs qualitatively and quantitatively, and discuss their pros and cons. Then we introduce our own three approaches of combing cycle-GANs and Gatys'CNN based model for Chinese painting style transfer.

Here are examples for "Gong-bi" style transfer and "Shui-mo" style transfer:
![gongbi](https://github.com/yanyangbaobeiIsEmma/ChinsePaintingStyleTransfer/blob/master/gongbiTransferExample.JPG)

![shuimo](https://github.com/yanyangbaobeiIsEmma/ChinsePaintingStyleTransfer/blob/master/shuimoTansferExample.jpg)
Try out our own approach on http://20.185.103.117:5000/ and have fun!

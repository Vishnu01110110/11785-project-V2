This pipeline integrates an Encoder-UNet-Decoder architecture for advanced image processing tasks. It begins with the Encoder, which compresses input images into a latent representation, followed by a UNet structure for feature extraction and transformation, and concludes with a Decoder that reconstructs images from the modified latent space. Designed for both image-to-image translation and generation from textual descriptions, this setup leverages depthwise separable convolutions to enhance performance and efficiency.

 How to run
1) Download v1-5-pruned-emaonly.ckpt from https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main add it to data 
2) download all requirments from requirements.txt
3) run demo for results

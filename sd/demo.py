import model_loader
import pipeline
from PIL import Image
import torch
import openai
import qrcode
import numpy as np
import cv2
from transformers import CLIPTokenizer

# API and device settings
api_key = 'KEY'
client = openai.OpenAI(api_key=api_key)
DEVICE = "cpu"
ALLOW_CUDA = False
ALLOW_MPS = False

# Paths
tokenizer_vocab_path = r"../data/tokenizer_vocab.json"
tokenizer_merges_path = r"../data/tokenizer_merges.txt"
model_file_path = r"../data/v1-5-pruned.ckpt"

if torch.cuda.is_available() and ALLOW_CUDA:
    DEVICE = "cpu"
elif (torch.has_mps or torch.backends.mps.is_available()) and ALLOW_MPS:
    DEVICE = "mps"
print(f"Using device: {DEVICE}")

tokenizer = CLIPTokenizer(tokenizer_vocab_path, merges_file=tokenizer_merges_path)
models = model_loader.preload_models_from_standard_weights(model_file_path, DEVICE)

def methodName(prompt, url):
    print("Type prompt" , type(prompt))
    print("Type URL" , type(url))
    #prompt = "A vibrant peacock displaying its spectacular tail feathers, which are fully fanned out, showing off the intricate eye patterns in a lush garden setting."
    #url = "https://www.linkedin.com/in/piyushtalreja-/"
    def get_image_generation_prompt(prompt):
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a summarizer. Your task is to summarize the given text for an image generation task. You need to summarize the text in less than 10 words adding words such as high resolution and artistic. Don't ask any follow up question. When in doubt just echo the text back."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content

    def qr_code_create(url):
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(url)
        qr.make(fit=True)
        img_qr = qr.make_image(fill_color="black", back_color="white").convert('RGB')
        return np.array(img_qr)

    def combine_images(qr_array, output_image_array, darken_factor=0.7):
        qrcode = np.array(Image.fromarray(qr_array).convert('RGB'))
        output_image = np.array(Image.fromarray(output_image_array).convert('RGB'))

        # Convert color channels from RGB to BGR for OpenCV
        qrcode = qrcode[:, :, ::-1]
        output_image = output_image[:, :, ::-1]

        # Resize QR code to match the output image size
        foreground_resized = cv2.resize(qrcode, (output_image.shape[1], output_image.shape[0]),
                                        interpolation=cv2.INTER_AREA)

        # Darken the background image
        output_image = (output_image * darken_factor).astype(np.uint8)

        # Create masks for combining images
        foreground_gray = cv2.cvtColor(foreground_resized, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(foreground_gray, 1, 255, cv2.THRESH_BINARY)
        inverse_mask = cv2.bitwise_not(mask)

        # Apply masks to separate foreground and background
        background_masked = cv2.bitwise_and(output_image, output_image, mask=inverse_mask)
        foreground_masked = cv2.bitwise_and(foreground_resized, foreground_resized, mask=mask)

        # Combine the darkened background with the original QR code
        combined = cv2.add(background_masked, foreground_masked)

        # Save the combined image to a file
        cv2.imwrite('combined_image.png', combined)

        return combined

    # Generate prompt and image
    refined_prompt = get_image_generation_prompt(prompt)
    print("REFINED PROMPT",refined_prompt)
    output_image = pipeline.random_seed_generator(
        prompt=refined_prompt,
        uncond_prompt="",
        input_image=None,
        strength=0.9,
        do_cfg=True,
        cfg_scale=8,
        sampler_name="ddpm",
        n_inference_steps=50,
        seed=None,
        models=models,
        device=DEVICE,
        idle_device="cpu",
        tokenizer=tokenizer,
    )

    output_image_array = Image.fromarray(output_image)
    qr_array = qr_code_create(url)
    combined_image = combine_images(qr_array, output_image)

    # Save and display the final image
    final_image = Image.fromarray(combined_image)
    final_image.save('combined_image.png')
    final_image.show()



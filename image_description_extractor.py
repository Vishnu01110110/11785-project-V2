import openai

def get_openai_response(prompt):
    openai.api_key = 'API-KEY'

    try:
        # Sending a request to the OpenAI API with the given prompt
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k-0613",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"An error occurred: {str(e)}"

def create_prompt():
    description = input("Enter your detailed image description: ")
    prompt_text = f"""
    You are participating in an image generation project. You will provide a detailed text description of an image you would like to generate. The focus of this task is to identify and extract key tokens from your description which will be used as input for an image generation model.

    Write a Detailed Description: {description}

    Extract Key Tokens: From your description, extract important words or phrases that are essential to the image. These tokens should represent the core components of the scene you envision and clearly associate each attribute with its corresponding subject.

    Format Your Submission: List the extracted tokens in a comma-separated format. Ensure that each token is a crucial element of your desired image and clearly assigned to the relevant subject.
    """
    return prompt_text



user_defined_prompt = create_prompt()
response = get_openai_response(user_defined_prompt)
print("API Response:", response)

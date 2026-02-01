# flake8: noqa
import os
import shutil
from eyepop import EyePopSdk
from eyepop.worker.worker_types import Pop, InferenceComponent
import json
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("EYEPOP_API_KEY")
print(f"Using API key: {api_key}")

objectOfInterest = 'Person'
questionList = (
    "Is there a person in the image (Yes/No), "
    "How much danger is the person in if they are in the fire (as a decimal from 0-1 (i.e 0.765) based on the criticality of the danger. 0 if no fire or no person)"
    "Report the values of the categories as classLabels. "
)

test_folder = './test/'  # Path to the test folder
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')  # Supported extensions
high_danger_folder = './highDanger/'  # Folder to save high-danger images
os.makedirs(high_danger_folder, exist_ok=True)

with EyePopSdk.workerEndpoint(
    api_key=api_key
) as endpoint:
    prompt = f"Analyze the image of {objectOfInterest} provided and determine the categories of: " + questionList + "If you are unable to provide a category with a value then set its classLabel to null"

    print(f"Using prompt: {prompt}")

    endpoint.set_pop(
       Pop(components=[
            InferenceComponent(
                id=1,
                ability='eyepop.image-contents:latest',
                params={"prompts": [
                            {
                                "prompt": prompt
                            }
                        ] }
            )
        ])
    )

    # Loop over all images in the test folder
    for filename in os.listdir(test_folder):
        if filename.lower().endswith(image_extensions):
            image_path = os.path.join(test_folder, filename)
            print(f"\nProcessing image: {image_path}")
            try:
                result = endpoint.upload(image_path).predict()
                print(json.dumps(result, indent=4))
                if "classes" in result and len(result["classes"]) > 1:
                    class_label = result['classes'][1]['classLabel']
                    print(f"Class label: {class_label}")
                    try:
                        label_value = float(class_label)
                        if label_value > 0.7:
                            shutil.copy(image_path, os.path.join(high_danger_folder, filename))
                            print(f"Saved to high danger folder: {filename}")
                    except ValueError:
                        print(f"Class label is not a number: {class_label}")
                else:
                    print("No class label found.")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
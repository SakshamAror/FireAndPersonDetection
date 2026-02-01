# flake8: noqa
import os
import subprocess
from eyepop import EyePopSdk
from eyepop.worker.worker_types import Pop, InferenceComponent
import json
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("EYEPOP_API_KEY")
print(f"Using API key: {api_key}")

example_image_path = './test/fire.661.png'
objectOfInterest = 'Person'
questionList = (
    "Is there a person in the image (Yes/No), "
    "How much danger is the person in if they are in the fire (as a decimal (2-3 decimal places) from 0-1 based on the criticality of the danger. 0 if no fire or no person)"
    "Report the values of the categories as classLabels. "
)

# Routes to YOLO model
def route_to_different_model(image_path):
    print("Running local model instead")
    # Run Yolo model here, inputting image_path

with EyePopSdk.workerEndpoint(
    api_key=api_key
) as endpoint:
    prompt = f"Analyze the image of {objectOfInterest} provided and determine the categories of: " + questionList + "If you are unable to provide a category with a value then set its classLabel to null"

    print (f"Using prompt: {prompt}")

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

    try:
        result = endpoint.upload(example_image_path).predict()
        print(json.dumps(result, indent=4))
        if "classes" in result and len(result["classes"]) > 1:
            print(result["classes"][1]['classLabel'])
        else:
            print("Class label not found, routing to different script.")
            route_to_different_model(example_image_path)
    except Exception as e:
        error_str = str(e).lower()
        if 'timeout' in error_str or 'overload' in error_str or 'server' in error_str or 'healthy' in error_str:
            print(f"Timeout or server overload error: {e}. Routing to different script.")
            route_to_different_model(example_image_path)
        else:
            print(f"Other error: {e}")

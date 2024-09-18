from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from llama_index.tools.wikipedia import WikipediaToolSpec
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.tools import FunctionTool
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
import json
from labels import all_animal_labels, dangerous_animal_labels

Settings.llm = OpenAI(model="gpt-4o-mini")

wiki_tool = WikipediaToolSpec()

def get_animal_info(animal_name):
    wiki_data = wiki_tool.load_data(animal_name)
    content = wiki_data.get("content", "")
    url = wiki_data.get("url", "")
    
    return {
        "summary": content[:500],  # take the first 500 characters as the summary
        "url": url
    }


def identify_image(image_path):
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    image = Image.open(image_path)

    inputs = clip_processor(text=all_animal_labels, images=image, return_tensors="pt", padding=True)
    outputs = clip_model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    
    animal_name = all_animal_labels[probs.argmax().item()]
    
    inputs = clip_processor(text=dangerous_animal_labels, images=image, return_tensors="pt", padding=True)
    outputs = clip_model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    
    # print(probs)
    is_dangerous = probs[0][0].item() > 0.5

    return animal_name, is_dangerous

identify_image_tool = FunctionTool.from_defaults(fn=identify_image)
get_animal_info_tool = FunctionTool.from_defaults(fn=get_animal_info)

agent = OpenAIAgent.from_tools(
    tools=[identify_image_tool, get_animal_info_tool],
    verbose=True
)


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnimalRecognition(BaseModel):
    animalName: str
    confidence: float
    description: str
    wikipediaUrl: str
    isDangerous: bool

@app.post("/recognize-animal")
async def recognize_animal(image: UploadFile = File(...)):
    # Save the uploaded image
    with open("temp_image.jpg", "wb") as buffer:
        buffer.write(await image.read())
    
    response = agent.chat("""Analyze this image: temp_image.jpg. Please respond in JSON format, including the following fields: 
    - animalName: the name of the animal, or "Unknown" if you can't identify it
    - confidence: a number between 0 and 1 indicating how confident you are in the identification
    - description: a brief description of the animal, or "No description available" if it's unknown
    - wikipediaUrl: the Wikipedia URL for the animal, or "No URL available" if it's unknown
    - isDangerous: boolean indicating if the animal is generally considered dangerous

    If you can't identify the animal or if it's not a specific animal (e.g., "some other animal" or "something else"), set animalName accordingly, set confidence to a low value, description to "No description available", and wikipediaUrl to "No URL available".""")
    
    # print(response)
    
    # parse the response
    response_text = str(response)
    json_start = response_text.rfind('```json')
    json_end = response_text.rfind('```')
    if json_start != -1 and json_end != -1:
        json_content = response_text[json_start+7:json_end].strip()
        animal_info = json.loads(json_content)

        animal_name = animal_info.get("animalName", "Unknown")
        confidence = animal_info.get("confidence", 0.5)  # use 0.5 as default confidence
        description = animal_info.get("description", "No description available")
        wikipedia_url = animal_info.get("wikipediaUrl", "No URL available")
        is_dangerous = animal_info.get("isDangerous", False)
         
        return AnimalRecognition(
            animalName=animal_name,
            confidence=confidence,
            description=description,
            wikipediaUrl=wikipedia_url,
            isDangerous=is_dangerous
        )
    else:
        raise ValueError("Unable to parse JSON content from the response")
    

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

Settings.llm = OpenAI(model="gpt-4o-mini")


wiki_tool = WikipediaToolSpec()

def get_animal_info(animal_name):
    wiki_content = wiki_tool.load_data(animal_name)
    return wiki_content[:500] 


def identify_image(image_path):
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    image = Image.open(image_path)

    # 這裡可以添加更多的標籤來識別具體的動物
    animal_labels = ["cat", "dog", "tiger", "elephant", "bird",  "fish", "Mola mola", "some other animal", "something else"]
    inputs = clip_processor(text=animal_labels, images=image, return_tensors="pt", padding=True)
    outputs = clip_model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    
    animal_name = animal_labels[probs.argmax().item()]
    
    inputs = clip_processor(text=["a dangerous animal", "a safe animal"], images=image, return_tensors="pt", padding=True)
    outputs = clip_model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    
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
    isDangerous: bool

@app.post("/recognize-animal")
async def recognize_animal(image: UploadFile = File(...)):
    # 保存上傳的圖片
    with open("temp_image.jpg", "wb") as buffer:
        buffer.write(await image.read())
    
    response = agent.chat("分析這張圖片：temp_image.jpg。回應格式為：動物名稱, 動物描述, 動物是否危險")
    
    print(response)
    
    # 返回模擬的識別結果
    return AnimalRecognition(
        animalName="Siberian Tiger (來自FastAPI)",
        confidence=0.95,
        description="The Siberian tiger is the largest living cat species and a member of the Panthera genus. It is the national animal of Russia and is classified as Endangered by the IUCN.",
        isDangerous=True
    )

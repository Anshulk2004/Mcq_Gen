import os
import json
import traceback
import pandas as pd
from dotenv import load_dotenv
from src.mcqgenerator.utils import read_file , get_table_data
from src.mcqgenerator.logger import logging


from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain


load_dotenv()
key = os.getenv("OPEN_API_KEY")

llm = ChatOpenAI(openai_api_key = key,model="gpt-3.5-turbo",temperature = 0.7)

template = """
Text:{text}
You are an expert MCQ Maker. Given the above text, it is your job to create a quiz of {number} multiple choice questions for {subject} students\
    in {tone} tone.Make sure the questions are not repeated and check all the questions to be conforming as well.The questions should be well made\
    Make sure to format  your response like RESPONSE_JSON below and use it as a guide.\
    Ensure to make {number} MCQs
    ### Response_JSON
    {response_json}
"""

Quiz_prompt =  PromptTemplate(
    input_variables=["text","number","subject","tone","response_json"],
    template=template
)

quiz_chain = LLMChain(llm=llm,prompt=Quiz_prompt,output_key="quiz",verbose=True)

template2 = """
You are an expert english grammarian and writer. Given a Multiple Choice Question for {subject} students.You need to evaluate the Complexity of the question \
    and give a complete analysis of the quiz. Only use at max 50 words for complexity. If the quiz is not at par with the cognitive and analytical abilities\
    of the students , update the quiz questions which needs to be changed and change the tone such that it perfectly fits the student abilities.
    Quiz_MCQs : 
    {quiz}
        
    Check from an expert english writer of the above quiz.
"""

Quiz_evaluation_prompt = PromptTemplate(input_variables=["subject","quiz"],template=template2)

Quiz_evaluation_chain = LLMChain(llm=llm,prompt=Quiz_evaluation_prompt,output_key='review',verbose=True)

generation_chain = SequentialChain(chains=[quiz_chain,Quiz_evaluation_chain],input_variables=["text","number","subject","tone","response_json"],
                                   output_variables=["quiz","review"],verbose=True)
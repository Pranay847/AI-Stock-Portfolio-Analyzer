import os
import joblib
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from rag.retrieval import query_index
from agents.prompts import DEFAULT_EXPLAIN_PROMPT

load_dotenv()


class Analyzer:
    def __init__(self, model_path='models/saved_models/xgb_model.pkl'):
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
        else:
            self.model = None
        self.llm = ChatOpenAI(temperature=0.2)

    def explain(self, context:str, prediction:str, confidence:float):
        prompt = PromptTemplate(template=DEFAULT_EXPLAIN_PROMPT, input_variables=['context','prediction','confidence'])
        chain = LLMChain(llm=self.llm, prompt=prompt)
        out = chain.run({'context': context, 'prediction': prediction, 'confidence': f"{confidence:.2f}"})
        return out

    def analyze_ticker(self, ticker:str, features_text:str, topk=3):
        # retrieve documents
        docs = query_index(query=f"Ticker: {ticker}", k=topk)
        context = '\n---\n'.join([d.page_content for d in docs])
        # placeholder prediction
        prediction = 'hold'
        confidence = 0.0
        if self.model is not None:
            try:
                # features_text should be a 1-row csv-like string; leave hooking to app
                pass
            except Exception:
                pass
        explanation = self.explain(context, prediction, confidence)
        return {
            'context': context,
            'prediction': prediction,
            'confidence': confidence,
            'explanation': explanation
        }

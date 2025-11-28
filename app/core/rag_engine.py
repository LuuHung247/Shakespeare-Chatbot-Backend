from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from app.core.llm_manager import llm_manager
from app.core.vector_store import vector_store_manager
from app.core.prompt_templates import SHAKESPEARE_POEM_TEMPLATE
from typing import Dict, List

class RAGEngine:
    """RAG Engine cho Shakespeare chatbot"""
    
    def __init__(self):
        self.llm = None
        self.vector_store = None
        self.qa_chain = None
        
    def initialize(self):
        """Khởi tạo RAG system"""
        # Load LLM
        self.llm = llm_manager.initialize()
        
        # Load vector store
        self.vector_store = vector_store_manager.load_vector_store()
        
        # Tạo prompt template
        prompt = PromptTemplate(
            template=SHAKESPEARE_POEM_TEMPLATE,
            input_variables=["context", "question"]
        )
        
        # Tạo QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_kwargs={"k": 5}
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        
        return self.qa_chain
    
    def generate_poem(self, user_input: str) -> Dict:
        """Generate poem với RAG"""
        if not self.qa_chain:
            self.initialize()
        
        try:
            # Retrieve relevant Shakespeare texts
            relevant_docs = vector_store_manager.similarity_search(
                user_input, 
                k=5
            )
            
            # Generate poem
            result = self.qa_chain.invoke({"query": user_input})
            
            return {
                "poem": result["result"],
                "sources": [
                    {
                        "content": doc.page_content[:200],
                        "metadata": doc.metadata
                    }
                    for doc in result.get("source_documents", [])
                ],
                "relevant_context": [doc.page_content for doc in relevant_docs]
            }
            
        except Exception as e:
            raise Exception(f"Error generating poem: {str(e)}")
    
    def chat(self, message: str, context: List[str] = None) -> str:
        """Chat mode - hội thoại với bot"""
        if not self.llm:
            self.llm = llm_manager.initialize()
        
        # Build prompt with context
        prompt = self._build_chat_prompt(message, context)
        
        try:
            response = llm_manager.generate(prompt)
            return response
        except Exception as e:
            raise Exception(f"Error in chat: {str(e)}")
    
    def _build_chat_prompt(self, message: str, context: List[str] = None) -> str:
        """Build prompt cho chat"""
        base_prompt = """You are a helpful assistant that speaks in Shakespearean style.
You help users create poems in the style of William Shakespeare.

"""
        if context:
            base_prompt += f"Previous conversation:\n{chr(10).join(context[-3:])}\n\n"
        
        base_prompt += f"User: {message}\nAssistant:"
        
        return base_prompt

# Singleton instance
rag_engine = RAGEngine()
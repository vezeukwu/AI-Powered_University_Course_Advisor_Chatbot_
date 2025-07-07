# AI-Powered_University_Course_Advisor_Chatbot_
In this project I developed an AI-powered course advisor chatbot for Teesside University UK using open-source tools. The chatbot provides real-time, document-grounded academic support by retrieving course and policy information from institutional documents through a Retrieval-Augmented Generation (RAG) framework.

## 🚀 Live Demo on Hugging Face Spaces
👉 [Try it out here](https://huggingface.co/spaces/vnwobodo/vnwobodo-demo)

Teesside University has its main campuses in Middlesbrough (TS1 3BX, North East England) and other campuses such as London campus in Stratford. The school's Motto: "Deeds Not Words" is shown in the outstanding performances and Awards received such as the TEF Gold 2023 for overall teaching and student experience; TEF Gold for teaching excellence (2023), Top in North East England f#or student experience Outstanding apprenticeships (Ofsted 2025) High student satisfaction. It offers Full-time and part-time study modes - Foundation & Pathway programmes, including Pre-Masters and writing-up routes Undergraduate Courses at Teesside University Teesside offers a wide selection of undergraduate courses
To enhance academic services and reduce manual workload, the university aims to implement an open-source AI chatbot that provides 24/7 student support

## Business Problems
- Advisors are overwhelmed with repetitive academic questions
- Students struggle to navigate course and regulation documents
- Lack of instant support leads to poor user experience
- Need for scalable, cost-effective automation

## Project Objectives:
- Provide instant academic guidance using AI-powered chatbot
- Retrieve accurate information from course-related documents
- Reduce staff burden through automation
- Host solution on Hugging Face platform
- Ensure open-source and privacy-friendly deployment

## Tech Stack & Tools
- Model: google/flan-t5-small via Hugging Face Transformers
- Embeddings: sentence-transformers/all-MiniLM-L6-v2
- Vector DB: FAISS
- Interface: Gradio
- Deployment Platform: Hugging Face Spaces

## Workflow Summary
- Extract details from the school's website, saving them in markdownfiles
- Used SentenceTransformers to generate text embeddings
- Stored vectors in FAISS for similarity search

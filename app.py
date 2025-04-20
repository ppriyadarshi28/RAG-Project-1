
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import faiss
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

def main():
    # Load environment variables from .env file
    load_dotenv()

    # Get the API key from the .env file
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        print("API key not found. Please set the OPENAI_API_KEY environment variable.")
    
    st.set_page_config(page_title="AE Literature Screening")

    st.header("Ask your AE Literature Chatbot 🤖", divider="gray", )

    # Initialize a list to store uploaded documents
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []

    # upload pdf file
    pdf = st.sidebar.file_uploader("Upload your literature document", type=["pdf"])

    # Add the uploaded file to the session state
    if pdf is not None:
        if pdf.name not in [file["name"] for file in st.session_state.uploaded_files]:
            st.session_state.uploaded_files.append({"name": pdf.name, "file": pdf})

    # Dropdown to select a document
    selected_file = st.sidebar.selectbox(
        "Select a document to process:",
        [file["name"] for file in st.session_state.uploaded_files])
    
    if "last_selected_file" not in st.session_state:
        st.session_state.last_selected_file = None

    if selected_file != st.session_state.last_selected_file:
        st.session_state.messages = []  # Clear chat history
        st.session_state.last_selected_file = selected_file  # Update the last selected file

    # Button to remove the selected document
    if selected_file and st.sidebar.button("Remove Selected Document"):
        # Remove the selected file from the session state
        st.session_state.uploaded_files = [
            file for file in st.session_state.uploaded_files if file["name"] != selected_file
        ]
        st.success(f"Document '{selected_file}' removed successfully!")
        st.rerun()  # Rerun the app to update the dropdown

    if selected_file:
        pdf = next(file["file"] for file in st.session_state.uploaded_files if file["name"] == selected_file)
        pdf_reader = PdfReader(pdf)
        pdf_text = ""
        for page in pdf_reader.pages:
            pdf_text += page.extract_text()

        # Display the "Summary" button
        if st.sidebar.button("Generate Summary"):
            # Predefined prompt for summarization
            summary_prompt = "Summarize the adverse drug reaction or adverse event or adverse reaction information present in the document in a concise manner:" \
            "\n\n" \
            "The summary should include the following information:\n" \
            "- The type of adverse event\n" \
            "- The drug involved\n" \
            "- The patient population affected\n" \
            "- The severity of the event\n" \
            "- The outcome of the event\n" \
            "- The reporting source\n" \
            "- Any other relevant information\n" \
            "\n" \
            "Please provide the summary in a structured format, such as table."

            # Combine the prompt with the document text
            input_text = f"{summary_prompt}\n\n{pdf_text}"
            
            # Use the OpenAI LLM to generate the summary
            llm = ChatOpenAI(api_key=api_key, model="gpt-3.5-turbo", temperature=0)
            response = llm.predict(input_text)
            
            # Display the summary in the main app
            st.subheader("Adverse Event Summary")
            st.write(response)
        
        # split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        chunks = text_splitter.split_text(pdf_text)

        # create embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)

        # create vector store
        knowledge_base = faiss.FAISS.from_texts(chunks, embeddings)

        # Initialize the chat message history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display the chat history on app load
        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar=message.get("avatar", None)):
                st.markdown(message["content"])

        # React to user input
        if user_question := st.chat_input("Ask a question about the literature document:"):
            # Display user message in chat message container
            with st.chat_message(name = "user", avatar="MyPic1.png"):
                st.markdown(user_question)
                st.session_state.messages.append({"role": "user", "content": user_question, "avatar": "MyPic1.png"})

        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            
            llm = ChatOpenAI(api_key=api_key, model="gpt-3.5-turbo", temperature=0)
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=user_question)

            # Display bot message in chat message container
            with st.chat_message("assistant"):
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
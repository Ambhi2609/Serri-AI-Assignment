import streamlit as st
import os
import tempfile
import asyncio
import logging
from pathlib import Path
import random

# Import your existing modules
from document_ingestion import DocumentIngestionPipeline, PipelineConfig
from support_agent import SupportBotAgent, AgentConfig

# ============================================================================
# PAGE CONFIG - MUST BE FIRST STREAMLIT COMMAND
# ============================================================================
st.set_page_config(
    page_title="Customer Support Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CONFIGURE LOGGING TO CENTRALIZED FILE
# ============================================================================
logging.getLogger().handlers.clear()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('support_bot_log.txt', mode='a'),
        logging.StreamHandler()
    ],
    force=True
)

logger = logging.getLogger(__name__)
logger.info("="*80)
logger.info("Streamlit Application Started")
logger.info("="*80)

# ============================================================================
# FORCE DARK MODE CSS
# ============================================================================
st.markdown("""
<style>
    .stApp {
        background-color: #0E1117 !important;
    }
    
    .stApp, .stApp * {
        color: #FFFFFF !important;
    }
    
    [data-testid="stSidebar"] {
        background-color: #262730 !important;
    }
    
    .stTextInput input, .stTextArea textarea, .stSelectbox select {
        background-color: #262730 !important;
        color: #FFFFFF !important;
        border: 1px solid #4A4A4A !important;
    }
    
    [data-testid="stFileUploader"] {
        background-color: #262730 !important;
    }
    
    .stButton button {
        background-color: #FF4B4B !important;
        color: #FFFFFF !important;
        border: none !important;
    }
    
    .stAlert {
        background-color: #262730 !important;
        color: #FFFFFF !important;
    }
    
    [data-testid="stExpander"] {
        background-color: #262730 !important;
        border: 1px solid #4A4A4A !important;
    }
    
    .stChatMessage {
        background-color: #262730 !important;
    }
    
    .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown h1, 
    .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        color: #FFFFFF !important;
    }
    
    code {
        background-color: #1E1E1E !important;
        color: #D4D4D4 !important;
        padding: 2px 4px !important;
        border-radius: 3px !important;
    }
    
    pre {
        background-color: #1E1E1E !important;
        color: #D4D4D4 !important;
        padding: 10px !important;
        border-radius: 5px !important;
    }
    
    /* Iteration boxes */
    .iteration-box {
        background-color: #1A1A2E !important;
        border-left: 4px solid #6C63FF !important;
        padding: 15px !important;
        margin: 10px 0 !important;
        border-radius: 5px !important;
    }
    
    .feedback-good {
        border-left-color: #28A745 !important;
        background-color: #1A3D2B !important;
    }
    
    .feedback-warning {
        border-left-color: #FFC107 !important;
        background-color: #3D3520 !important;
    }
    
    .feedback-error {
        border-left-color: #DC3545 !important;
        background-color: #3D1E22 !important;
    }
    
    /* Selectbox styling */
    div[data-baseweb="select"] {
        background-color: #262730 !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# INITIALIZE SESSION STATE
# ============================================================================
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'document_processed' not in st.session_state:
    st.session_state.document_processed = False
if 'api_key_valid' not in st.session_state:
    st.session_state.api_key_valid = False
if 'document_name' not in st.session_state:
    st.session_state.document_name = None
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "openrouter/sherlock-dash-alpha"

# ============================================================================
# FREE LLM OPTIONS
# ============================================================================
FREE_LLMS = {
    "Sherlock Dash Alpha": "openrouter/sherlock-dash-alpha",
    "Mistral 7B Instruct": "mistralai/mistral-7b-instruct:free",
    "GPT OSS 20B": "openai/gpt-oss-20b:free",
    "Qwen 2.5 72B Instruct": "qwen/qwen-2.5-72b-instruct:free",
    "Llama 3.3 70B Instruct": "meta-llama/llama-3.3-70b-instruct:free"
}

# ============================================================================
# HEADER
# ============================================================================
st.markdown('<h1 style="color: #FFFFFF !important;">🤖 Customer Support Bot</h1>', unsafe_allow_html=True)
st.markdown('<p style="color: #CCCCCC !important;">Intelligent document-based Q&A with random feedback & answer refinement</p>', unsafe_allow_html=True)
st.markdown("---")

# ============================================================================
# SIDEBAR - CONFIGURATION & DOCUMENT UPLOAD
# ============================================================================
with st.sidebar:
    st.markdown('<h2 style="color: #FFFFFF !important;">⚙️ Configuration</h2>', unsafe_allow_html=True)
    
    # ============================================================================
    # LLM SELECTION
    # ============================================================================
    st.markdown('<h3 style="color: #FFFFFF !important;">🤖 Select LLM</h3>', unsafe_allow_html=True)
    selected_model_name = st.selectbox(
        "Choose a free OpenRouter model",
        options=list(FREE_LLMS.keys()),
        index=list(FREE_LLMS.values()).index(st.session_state.selected_model),
        label_visibility="collapsed"
    )
    st.session_state.selected_model = FREE_LLMS[selected_model_name]
    st.info(f"📌 Using: **{selected_model_name}**")
    
    st.markdown("---")
    
    # ============================================================================
    # API KEY INPUT
    # ============================================================================
    st.markdown('<h3 style="color: #FFFFFF !important;">🔑 OpenRouter API Key</h3>', unsafe_allow_html=True)
    api_key = st.text_input(
        "Enter your OpenRouter API key",
        type="password",
        value=os.getenv("OPENROUTER_API_KEY", ""),
        label_visibility="collapsed"
    )
    
    if api_key:
        st.session_state.api_key_valid = True
        st.success("✅ API Key loaded")
        logger.info("API Key validated")
    else:
        st.warning("⚠️ Please enter API key")
    
    st.markdown("---")
    
    # ============================================================================
    # CACHE CONFIGURATION
    # ============================================================================
    st.markdown('<h3 style="color: #FFFFFF !important;">💾 Semantic Cache</h3>', unsafe_allow_html=True)
    
    enable_cache = st.checkbox(
        "Enable Semantic Caching",
        value=True,
        help="Cache similar queries to save time and tokens"
    )
    
    if enable_cache:
        cache_threshold = st.slider(
            "Cache Similarity Threshold",
            min_value=0.80,
            max_value=0.99,
            value=0.90,
            step=0.01,
            help="Minimum similarity score for cache hit (0.90 = 90% similar)"
        )
        st.info(f"🎯 Threshold: {cache_threshold:.0%}")
    else:
        cache_threshold = 0.90
    
    st.markdown("---")
    
    # ============================================================================
    # DOCUMENT UPLOAD SECTION
    # ============================================================================
    st.markdown('<h3 style="color: #FFFFFF !important;">📄 Document Upload</h3>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload PDF document",
        type=['pdf'],
        label_visibility="collapsed"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        chunk_size = st.number_input("Chunk Size", value=500, min_value=100, max_value=2000, step=100)
    with col2:
        overlap = st.number_input("Overlap", value=120, min_value=0, max_value=500, step=10)
    
    window_size = st.slider("Context Window", min_value=0, max_value=5, value=3, 
                           help="Number of surrounding chunks to include for context")
    
    process_button = st.button("🔄 Process Document", use_container_width=True)
    
    # ============================================================================
    # DOCUMENT PROCESSING
    # ============================================================================
    if process_button and uploaded_file and st.session_state.api_key_valid:
        with st.spinner("Processing document..."):
            try:
                logger.info(f"Starting document processing: {uploaded_file.name}")
                
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name
                
                # Initialize pipeline
                config = PipelineConfig(
                    chunk_size=chunk_size,
                    chunk_overlap=overlap,
                    embedding_model='sentence-transformers/all-MiniLM-L6-v2',
                    similarity_threshold=0.3
                )
                
                st.session_state.pipeline = DocumentIngestionPipeline(config=config)
                
                # Ingest document
                num_chunks = st.session_state.pipeline.ingest_document(
                    pdf_path=tmp_path,
                    metadata={'filename': uploaded_file.name},
                    window_size=window_size
                )
                
                # Initialize agent with selected model and cache settings
                agent_config = AgentConfig(
                    answer_model=st.session_state.selected_model,
                    similarity_threshold=0.3,
                    max_context_length=15000
                )
                
                st.session_state.agent = SupportBotAgent(
                    pipeline=st.session_state.pipeline,
                    openrouter_api_key=api_key,
                    config=agent_config,
                    enable_cache=enable_cache,
                    cache_threshold=cache_threshold
                )
                
                # Cleanup
                os.unlink(tmp_path)
                
                st.session_state.document_processed = True
                st.session_state.document_name = uploaded_file.name
                st.success(f"✅ Processed {num_chunks} chunks from {uploaded_file.name}")
                logger.info(f"Document processing completed: {num_chunks} chunks created")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                logger.error(f"Document processing error: {str(e)}", exc_info=True)
    
    st.markdown("---")
    
    # ============================================================================
    # STATISTICS SECTION
    # ============================================================================
    if st.session_state.agent:
        st.markdown('<h3 style="color: #FFFFFF !important;">📊 Statistics</h3>', unsafe_allow_html=True)
        
        # Get stats
        stats = st.session_state.agent.get_stats()
        
        # Query Stats
        st.markdown("**Query Statistics**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Queries", stats.total_queries_processed)
            st.metric("In Scope", stats.in_scope_count)
        with col2:
            st.metric("Out of Scope", stats.out_of_scope_count)
            st.metric("Avg Time (s)", f"{stats.average_processing_time:.2f}")
        
        # Cache Stats (if enabled)
        if st.session_state.agent.cache:
            st.markdown("---")
            st.markdown("**Cache Statistics**")
            cache_stats = st.session_state.agent.cache.get_stats()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Cache Hits", cache_stats['hits'], help="Queries served from cache")
                st.metric("Cache Misses", cache_stats['misses'], help="Queries not found in cache")
            with col2:
                st.metric("Cached Entries", cache_stats['total_cached'], help="Total queries stored")
                st.metric("Hit Rate", f"{cache_stats['hit_rate']:.1f}%", help="Percentage of cache hits")
            
            # Cache threshold info
            st.info(f"🎯 Similarity Threshold: {cache_stats['threshold']:.0%}")
            
            # Clear cache button
            if st.button("🗑️ Clear Cache", use_container_width=True, help="Remove all cached queries"):
                st.session_state.agent.cache.clear()
                st.success("✅ Cache cleared successfully!")
                st.rerun()
        else:
            st.markdown("---")
            st.warning("💾 Semantic caching is disabled")
    
    # ============================================================================
    # HELP SECTION
    # ============================================================================
    st.markdown("---")
    st.markdown('<h3 style="color: #FFFFFF !important;">ℹ️ Help</h3>', unsafe_allow_html=True)
    
    with st.expander("About Semantic Cache"):
        st.markdown("""
        **Semantic caching** stores query responses and retrieves them for similar future queries.
        
        **Benefits:**
        - ⚡ Instant responses for similar queries
        - 💰 Saves API tokens and costs
        - 📊 Improves user experience
        
        **How it works:**
        1. Query embeddings are stored in FAISS
        2. Similar queries (above threshold) return cached responses
        3. Cache persists across sessions
        
        **Threshold Guide:**
        - 0.85-0.89: More cache hits, less precision
        - 0.90-0.94: Balanced (recommended)
        - 0.95-0.99: High precision, fewer hits
        """)
    
    with st.expander("About LLM Models"):
        st.markdown("""
        All models are **free** on OpenRouter:
        
        - **Sherlock Dash Alpha**: Fast, experimental
        - **Mistral 7B**: Balanced performance
        - **GPT OSS 20B**: Strong reasoning
        - **Qwen 3 4B**: Lightweight, fast
        - **Qwen 2.5 72B**: High quality answers
        - **Llama 3.3 70B**: Latest Meta model
        """)
    
    with st.expander("Chunking Parameters"):
        st.markdown("""
        **Chunk Size**: Number of characters per chunk
        - Smaller (100-300): More precise retrieval
        - Larger (500-1000): More context per chunk
        
        **Overlap**: Overlapping characters between chunks
        - Prevents information loss at boundaries
        - Recommended: 20-30% of chunk size
        
        **Context Window**: Neighboring chunks to include
        - 0: Only matched chunk
        - 3-5: More surrounding context (recommended)
        """)

# ============================================================================
# MAIN CHAT INTERFACE
# ============================================================================
st.markdown('<h2 style="color: #FFFFFF !important;">💬 Chat</h2>', unsafe_allow_html=True)

if st.session_state.document_processed:
    st.info(f"📄 Document: **{st.session_state.document_name}** | 🤖 Model: **{selected_model_name}**")
else:
    st.warning("⚠️ Please upload and process a document first")

# Display chat history with sequential answer progression
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # For user messages, just show the message
        if message["role"] == "user":
            st.markdown(f'<div style="color: #FFFFFF !important;">{message["content"]}</div>', unsafe_allow_html=True)
        
        # For assistant messages, show iteration progression
        else:
            st.markdown(f'<div style="color: #FFFFFF !important;">{message["content"]}</div>', unsafe_allow_html=True)
            
            # Display iteration progression
            if "iteration_details" in message and message["iteration_details"]:
                st.markdown("---")
                st.markdown("### 🔄 Answer Evolution")
                
                for idx, iter_detail in enumerate(message["iteration_details"], 1):
                    answer = iter_detail["answer"]
                    feedback = iter_detail["feedback"]
                    
                    # Display Answer
                    st.markdown(f"""
                    <div style="background-color: #1A1A2E; padding: 15px; border-radius: 8px; margin: 15px 0; border-left: 4px solid #4A9EFF;">
                        <strong style="color: #4A9EFF !important;">📝 Answer {idx}</strong><br><br>
                        <div style="color: #FFFFFF !important; line-height: 1.6;">
                            {answer}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display Feedback (if not the last iteration)
                    if idx < len(message["iteration_details"]):
                        if feedback["type"] == "good":
                            emoji = "✅"
                            color = "#28A745"
                        elif feedback["type"] == "too_vague":
                            emoji = "⚠️"
                            color = "#FFC107"
                        else:
                            emoji = "❌"
                            color = "#DC3545"
                        
                        st.markdown(f"""
                        <div style="background-color: rgba(0,0,0,0.3); padding: 12px; border-radius: 8px; margin: 10px 0 10px 20px; border-left: 3px solid {color};">
                            <strong style="color: {color} !important;">{emoji} Feedback {idx}</strong><br>
                            <strong style="color: #CCCCCC !important;">Type:</strong> <span style="color: {color} !important;">{feedback["type"]}</span><br>
                            <strong style="color: #CCCCCC !important;">Comment:</strong> {feedback["comment"]}<br>
                            <strong style="color: #CCCCCC !important;">Confidence:</strong> {feedback["confidence"]:.1%}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Arrow pointing to next iteration
                        if idx < len(message["iteration_details"]):
                            st.markdown("""
                            <div style="text-align: center; margin: 10px 0;">
                                <span style="color: #888888; font-size: 24px;">⬇️</span>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # For the last iteration, show final feedback
                    else:
                        if feedback["type"] == "good":
                            st.success(f"✅ Final feedback: {feedback['comment']}")
                        else:
                            st.info(f"ℹ️ Final feedback: {feedback['comment']} (Max iterations reached)")
# ============================================================================
# CHAT INPUT HANDLER
# ============================================================================
if prompt := st.chat_input("Ask a question about your document...", disabled=not st.session_state.document_processed):
    logger.info(f"New user query: {prompt}")
    
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(f'<div style="color: #FFFFFF !important;">{prompt}</div>', unsafe_allow_html=True)
    
    # Generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        iterations_placeholder = st.empty()
        
        # Update agent config with currently selected model
        st.session_state.agent.config.answer_model = st.session_state.selected_model
        
        # Check cache first
        cached_result = None
        if st.session_state.agent.cache:
            with st.spinner("🔍 Checking cache..."):
                cached_result = st.session_state.agent.cache.get(prompt)
        
        # ========================================================================
        # HANDLE CACHED RESPONSE - WITH FIXED STATS TRACKING
        # ========================================================================
        if cached_result:
            logger.info(f"✅ Cache HIT - Similarity: {cached_result['similarity_score']:.4f}")
            
            # ✅ FIX: Increment cache statistics manually
            if hasattr(st.session_state.agent.cache, 'hits'):
                st.session_state.agent.cache.hits += 1
            
            # Update agent internal stats
            if hasattr(st.session_state.agent, '_stats'):
                st.session_state.agent._stats['total_queries'] = st.session_state.agent._stats.get('total_queries', 0) + 1
                st.session_state.agent._stats['in_scope'] = st.session_state.agent._stats.get('in_scope', 0) + 1
            
            # Display cache hit banner
            message_placeholder.markdown(f"""
            <div style="background-color: #1A3D2B; padding: 12px; border-radius: 8px; margin-bottom: 15px; border-left: 4px solid #28A745;">
                <strong style="color: #28A745 !important;">🚀 Cached Response</strong><br>
                <span style="color: #CCCCCC !important;">Similarity: {cached_result['similarity_score']:.2%} | 
                Cached at: {cached_result['cached_at'][:19]}</span>
            </div>
            <div style="color: #FFFFFF !important;"><strong>Final Answer:</strong><br>{cached_result["answer"]}</div>
            """, unsafe_allow_html=True)
            
            # Display cached iteration details
            if cached_result['iteration_details']:
                with iterations_placeholder.container():
                    st.markdown("---")
                    st.markdown("### 📝 Answer Evolution (From Cache)")
                    
                    for idx, iter_detail in enumerate(cached_result['iteration_details'], 1):
                        answer = iter_detail["answer"]
                        feedback = iter_detail["feedback"]
                        
                        # Display Answer Box
                        st.markdown(f"""
                        <div style="background-color: #1A1A2E; padding: 15px; border-radius: 8px; margin: 15px 0; border-left: 4px solid #4A9EFF;">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                                <strong style="color: #4A9EFF !important;">📄 Answer {idx}</strong>
                            </div>
                            <div style="color: #FFFFFF !important; line-height: 1.6;">
                                {answer}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Display Feedback (if not last iteration)
                        if idx < len(cached_result['iteration_details']):
                            if feedback["type"] == "good":
                                emoji = "🟢"
                                color = "#28A745"
                            elif feedback["type"] == "too_vague":
                                emoji = "🟡"
                                color = "#FFC107"
                            else:
                                emoji = "🔴"
                                color = "#DC3545"
                            
                            st.markdown(f"""
                            <div style="background-color: rgba(0,0,0,0.3); padding: 12px; border-radius: 8px; margin: 10px 0 10px 20px; border-left: 3px solid {color};">
                                <strong style="color: {color} !important;">{emoji} Feedback {idx}</strong><br>
                                <strong style="color: #CCCCCC !important;">Type:</strong> <span style="color: {color} !important;">{feedback["type"]}</span><br>
                                <strong style="color: #CCCCCC !important;">Comment:</strong> {feedback["comment"]}<br>
                                <strong style="color: #CCCCCC !important;">Confidence:</strong> {feedback["confidence"]:.1%}
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Arrow to next iteration
                            st.markdown("""
                            <div style="text-align: center; margin: 10px 0;">
                                <span style="color: #888888; font-size: 24px;">⬇️</span>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            if feedback["type"] == "good":
                                st.success(f"✅ Final feedback: {feedback['comment']}")
                            else:
                                st.info(f"ℹ️ Final feedback: {feedback['comment']} (Max iterations reached)")
            
            # Save to session state with cache flag
            st.session_state.messages.append({
                "role": "assistant",
                "content": cached_result['answer'],
                "iteration_details": cached_result['iteration_details'],
                "from_cache": True,
                "similarity_score": cached_result['similarity_score']
            })
        
        # ========================================================================
        # GENERATE NEW RESPONSE
        # ========================================================================
        else:
            with st.spinner("🤖 Generating answer..."):
                try:
                    # Run query with max_iterations=2 (generates 3 total answers: 0, 1, 2)
                    response = asyncio.run(
                        st.session_state.agent.answer_query(prompt, max_iterations=2)
                    )
                    
                    # Log details
                    logger.info("="*80)
                    logger.info(f"QUERY: {prompt}")
                    logger.info(f"MODEL: {st.session_state.selected_model}")
                    logger.info(f"FINAL ANSWER: {response.answer}")
                    logger.info(f"ITERATIONS: {response.iterations}")
                    logger.info(f"IN SCOPE: {response.in_scope}")
                    logger.info("="*80)
                    
                    # ================================================================
                    # HANDLE OUT-OF-SCOPE QUERY
                    # ================================================================
                    if not response.in_scope:
                        message_placeholder.markdown(f"""
                        <div style="background-color: #3D1E22; padding: 12px; border-radius: 8px; margin-bottom: 15px; border-left: 4px solid #DC3545;">
                            <strong style="color: #DC3545 !important;">❌ Out of Scope</strong><br>
                            <span style="color: #CCCCCC !important;">This query is not covered by the uploaded document.</span>
                        </div>
                        <div style="color: #FFFFFF !important;">{response.answer}</div>
                        """, unsafe_allow_html=True)
                        
                        # Save to session state
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response.answer,
                            "iteration_details": [],
                            "from_cache": False,
                            "in_scope": False
                        })
                        
                        logger.info("Query was out of scope")
                    
                    # ================================================================
                    # HANDLE IN-SCOPE QUERY - DISPLAY ALL 3 ANSWERS + 2 FEEDBACKS
                    # ================================================================
                    else:
                        # Prepare iteration details
                        iteration_details_list = []
                        for iter_detail in response.iteration_details:
                            iteration_details_list.append({
                                "iteration_number": iter_detail.iteration_number,
                                "answer": iter_detail.answer,
                                "feedback": {
                                    "type": iter_detail.feedback.type,
                                    "comment": iter_detail.feedback.comment,
                                    "confidence": iter_detail.feedback.confidence
                                },
                                "tokens_used": iter_detail.tokens_used
                            })
                        
                        # Display all iterations FIRST
                        with iterations_placeholder.container():
                            st.markdown("---")
                            st.markdown("### 📝 Answer Evolution")
                            
                            for idx, iter_detail in enumerate(iteration_details_list, 1):
                                answer = iter_detail["answer"]
                                feedback = iter_detail["feedback"]
                                tokens = iter_detail.get("tokens_used", 0)
                                
                                # Display Answer Box
                                st.markdown(f"""
                                <div style="background-color: #1A1A2E; padding: 15px; border-radius: 8px; margin: 15px 0; border-left: 4px solid #4A9EFF;">
                                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                                        <strong style="color: #4A9EFF !important;">📄 Answer {idx}</strong>
                                        <span style="color: #888888; font-size: 12px;">💬 {tokens} tokens</span>
                                    </div>
                                    <div style="color: #FFFFFF !important; line-height: 1.6;">
                                        {answer}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Display Feedback (if not last answer)
                                if idx < len(iteration_details_list):
                                    if feedback["type"] == "good":
                                        emoji = "🟢"
                                        color = "#28A745"
                                    elif feedback["type"] == "too_vague":
                                        emoji = "🟡"
                                        color = "#FFC107"
                                    else:
                                        emoji = "🔴"
                                        color = "#DC3545"
                                    
                                    st.markdown(f"""
                                    <div style="background-color: rgba(0,0,0,0.3); padding: 12px; border-radius: 8px; margin: 10px 0 10px 20px; border-left: 3px solid {color};">
                                        <strong style="color: {color} !important;">{emoji} Feedback {idx}</strong><br>
                                        <strong style="color: #CCCCCC !important;">Type:</strong> <span style="color: {color} !important;">{feedback["type"]}</span><br>
                                        <strong style="color: #CCCCCC !important;">Comment:</strong> {feedback["comment"]}<br>
                                        <strong style="color: #CCCCCC !important;">Confidence:</strong> {feedback["confidence"]:.1%}
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # Arrow to next iteration
                                    st.markdown("""
                                    <div style="text-align: center; margin: 10px 0;">
                                        <span style="color: #888888; font-size: 24px;">⬇️</span>
                                    </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    # Show final feedback status
                                    if feedback["type"] == "good":
                                        st.success(f"✅ Final feedback: {feedback['comment']}")
                                    else:
                                        st.info(f"ℹ️ Final feedback: {feedback['comment']} (Max iterations reached)")
                        
                        # Display summary banner at TOP (FIXED HTML rendering)
                        message_placeholder.markdown(f"""
                        <div style="background-color: #1A3D2B; padding: 12px; border-radius: 8px; margin-bottom: 15px; border-left: 4px solid #28A745;">
                            <strong style="color: #28A745 !important;">✅ Response Generated</strong><br>
                            <span style="color: #CCCCCC !important;">Model: {st.session_state.selected_model} | 
                            Iterations: {response.iterations} | 
                            Time: {response.processing_time:.2f}s</span>
                        </div>
                        <div style="color: #FFFFFF !important;"><strong>Final Answer:</strong><br>{response.answer}</div>
                        """, unsafe_allow_html=True)
                        
                        # Save to session state
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response.answer,
                            "iteration_details": iteration_details_list,
                            "from_cache": False,
                            "in_scope": True,
                            "processing_time": response.processing_time,
                            "iterations": response.iterations
                        })
                        
                        logger.info("✅ Response generated and cached successfully")
                
                except Exception as e:
                    error_msg = f"❌ Error generating response: {str(e)}"
                    message_placeholder.error(error_msg)
                    logger.error(error_msg, exc_info=True)
                    
                    # Save error to session state
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "iteration_details": [],
                        "from_cache": False,
                        "error": True
                    })



# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #888888 !important;">
    <p style="color: #888888 !important;">Built with Streamlit • Powered by OpenRouter • Using {selected_model_name}</p>
    <p style="color: #888888 !important;">📄 Check <code>support_bot_log.txt</code> for detailed logs</p>
</div>
""", unsafe_allow_html=True)

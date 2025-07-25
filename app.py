# AI-Powered Excel Mock Interviewer - Complete Enhanced Version
import streamlit as st
import os
from interview.chains import InterviewManager
from interview.reports import generate_pdf_report
import threading
import time
from email_validator import validate_email, EmailNotValidError
from pathlib import Path
import json
from openai import OpenAI        
from dotenv import load_dotenv
load_dotenv()


# Configure page
st.set_page_config(
    page_title="Excel Mock Interviewer",
    page_icon="📊",
    layout="wide"
)

def initialize_session_state():
    """Initialize session state variables"""
    defaults = {
        'interview_state': 'setup',
        'interview': None,
        'candidate_name': "",
        'candidate_email': "",
        'current_question_index': 0,
        'chat_history': [],
        'answer_key': 0,  # For clearing answer box
        'timer_started': False,
        'start_time': None,
        'admin_mode': False,
        'clarification_history': []
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def validate_email_format(email):
    """Validate email format with proper domain checking"""
    try:
        validated = validate_email(email)
        email_lower = validated.email.lower()
        
        # Check for common valid domains
        valid_domains = [
            'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'live.com',
            'icloud.com', 'protonmail.com', 'aol.com', 'mail.com', 'zoho.com'
        ]
        
        domain = email_lower.split('@')[-1]
        
        # Allow company domains (contains a dot and is reasonable length)
        if '.' in domain and len(domain.split('.')) >= 2:
            domain_parts = domain.split('.')
            if len(domain_parts[-1]) >= 2 and len(domain_parts[-2]) >= 2:
                return True
        
        # Check against common domains
        if domain in valid_domains:
            return True
            
        return False
        
    except EmailNotValidError:
        return False


def show_timer():
    """Display countdown timer"""
    if st.session_state.interview and st.session_state.interview.timer_started:
        remaining = st.session_state.interview.get_remaining_time()
        minutes = remaining // 60
        seconds = remaining % 60
        
        if remaining <= 300:  # Last 5 minutes - show warning
            st.error(f"⏰ **Time Remaining: {minutes:02d}:{seconds:02d}**")
        elif remaining <= 600:  # Last 10 minutes - show caution
            st.warning(f"⏰ **Time Remaining: {minutes:02d}:{seconds:02d}**")
        else:
            st.info(f"⏰ **Time Remaining: {minutes:02d}:{seconds:02d}**")
        
        # Auto-submit if time is up
        if remaining <= 0 and not st.session_state.interview.is_interview_complete():
            st.session_state.interview.auto_submit_interview()
            st.session_state.interview_state = 'completed'
            st.error("⏰ **Time's up! Interview automatically submitted.**")
            st.rerun()

def show_admin_dashboard():
    """Show admin-only report management"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("👨‍💼 Admin Dashboard")
    
    reports_dir = Path("reports")
    if reports_dir.exists():
        report_files = list(reports_dir.glob("*.pdf"))
        
        if report_files:
            st.sidebar.write(f"📊 **Available Reports:** {len(report_files)}")
            
            for report_file in sorted(report_files, key=lambda x: x.stat().st_mtime, reverse=True):
                # Extract candidate info from filename
                parts = report_file.stem.replace('interview_report_', '').split('_')
                candidate_name = parts[0] if parts else "Unknown"
                
                with st.sidebar.expander(f"📄 {candidate_name}"):
                    st.write(f"**File:** {report_file.name}")
                    st.write(f"**Modified:** {time.ctime(report_file.stat().st_mtime)}")
                    
                    with open(report_file, "rb") as f:
                        st.download_button(
                            label="📥 Download Report",
                            data=f.read(),
                            file_name=report_file.name,
                            mime="application/pdf",
                            key=f"admin_download_{report_file.stem}"
                        )
        else:
            st.sidebar.info("No reports available yet.")
    else:
        st.sidebar.info("Reports directory not found.")

def show_clarification_bot():
    """Show instant Q&A clarification bot"""
    with st.expander("🤔 Need clarification? Ask here!", expanded=False):
        clarification_question = st.text_input(
            "Your question:",
            placeholder="e.g., 'Experiencing problems with question loading.'",
            key="clarification_input"
        )
        
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("Ask Bot", key="clarification_ask"):
                q = clarification_question.strip()
                if not q:
                    st.error("❌ Please enter a question.")
                elif not st.session_state.interview:
                    st.error("❌ Interview not started yet.")
                else:
                    reply = st.session_state.interview.ask_clarification(q)
                    st.success("✅ Bot replied below.")
                    st.session_state.clarification_history.append(
                        {"question": q, "response": reply}
                    )

        
        # Show recent clarifications
        if st.session_state.clarification_history:
            st.markdown("**Recent Clarifications:**")
            for i, qa in enumerate(reversed(st.session_state.clarification_history[-3:])):
                with st.container():
                    st.markdown(f"**Q:** {qa['question']}")
                    st.markdown(f"**A:** {qa['response']}")
                    if i < len(st.session_state.clarification_history[-3:]) - 1:
                        st.markdown("---")

def main():
    initialize_session_state()
    
    st.title("📊 AI-Powered Excel Mock Interviewer")
    st.markdown("*Enhanced with Open-Source AI and Advanced Features*")
    st.markdown("---")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Admin mode toggle
        admin_password = st.text_input("Admin Password", type="password", help="Enter admin password for report access")
        if admin_password == "admin123":  # Simple admin auth - replace with proper auth in production
            st.session_state.admin_mode = True
            st.success("✅ Admin mode activated")
        
        # API Key input (optional now with local embeddings)
        api_key = st.text_input(
            "OpenAI API Key (Optional)", 
            type="password", 
            value="",
            help="Enter your own OpenAI key to override the default GroqCloud integration."
        )
        
        # Model selection
        model = st.selectbox(
            "LLM Model",
            ["meta-llama/llama-4-scout-17b-16e-instruct","gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
            index=0,
            help="Used only if you provide an OpenAI API key."
        )
        # Timer display
        if st.session_state.interview_state == 'active':
            show_timer()
        
        st.markdown("---")
        
        
        # Session controls
        st.subheader("📋 Session Controls")
        
        # Regular download for completed interviews
        if (st.session_state.interview and 
            st.session_state.interview_state == 'completed' and
            not st.session_state.admin_mode):
            if st.button("📥 Download My Report"):
                try:
                    report_data = generate_pdf_report(st.session_state.interview.get_session_data())
                    st.download_button(
                        label="📄 Download PDF Report",
                        data=report_data,
                        file_name=f"interview_report_{st.session_state.candidate_name.replace(' ', '_')}.pdf",
                        mime="application/pdf"
                    )
                except Exception as e:
                    st.error(f"Error generating report: {e}")
        
        # Admin dashboard
        if st.session_state.admin_mode:
            show_admin_dashboard()
            ALERTS_LOG = Path("data/alerts.json")
            st.sidebar.subheader("⚠️ Live Support Alerts")

            if ALERTS_LOG.exists():
                lines = ALERTS_LOG.read_text(encoding="utf-8").splitlines()
                if lines:
                    # show the most recent 5 alerts
                    for line in lines[-5:]:
                        alert = json.loads(line)
                        st.sidebar.error(
                            f"🚨 {alert['candidate_name']} ({alert['candidate_email']})\n"
                            f"Issue: {alert['question']}\n"
                            f"At: {alert['timestamp']}"
                        )
                else:
                    st.sidebar.info("No active tech-support tickets.")
            else:
                st.sidebar.info("No support alerts logged yet.")

        
        if st.button("🔄 Reset Session"):
            # Clear all session state
            for key in list(st.session_state.keys()):
                if key not in ['admin_mode']:  # Preserve admin mode
                    del st.session_state[key]
            st.rerun()

    # Main content area
    
    # State: Setup (Collect candidate information)
    if st.session_state.interview_state == 'setup':
        st.header("👤 Candidate Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            candidate_name = st.text_input(
                "Full Name *",
                value=st.session_state.candidate_name,
                placeholder="Enter your full name",
                key="name_input"
            )
        
        with col2:
            candidate_email = st.text_input(
                "Email Address *",
                value=st.session_state.candidate_email,
                placeholder="user@example.com",
                key="email_input"
            )
        
        # Email validation feedback
        if candidate_email and not validate_email_format(candidate_email):
            st.error("❌ Please enter a valid email address with a proper domain (e.g., @gmail.com, @company.com)")
        
        st.markdown("---")
        
        if st.button("📋 View Interview Instructions", use_container_width=True):
            if candidate_name.strip() and candidate_email.strip():
                if validate_email_format(candidate_email):
                    st.session_state.candidate_name = candidate_name.strip()
                    st.session_state.candidate_email = candidate_email.strip()
                    st.session_state.interview_state = 'instructions'
                    st.rerun()
                else:
                    st.error("❌ Please enter a valid email address.")
            else:
                st.error("❌ Please fill in all required fields.")
    
    # State: Instructions
    elif st.session_state.interview_state == 'instructions':
        st.header("📋 Interview Instructions")
        
        st.markdown(f"""
        **Welcome, {st.session_state.candidate_name}!**
        
        You are about to take an Excel Skills Assessment. Please read the following instructions carefully:
        
        ### 📝 Interview Format
        - **Duration**: **30 minutes** (auto-submit when time expires)
        - **Questions**: 8 questions covering different Excel topics
        - **Difficulty Levels**: Basic → Intermediate → Advanced
        - **Question Types**: Text-based, image-supported, and practical scenarios
        
        ### 🎯 What to Expect
        1. **Timer**: Visible countdown in the sidebar - interview auto-submits at 00:00
        2. **Progressive Difficulty**: Questions adapt based on your performance
        3. **Clarification Support**: Use the "Need clarification?" section if confused
        4. **Various Formats**: Some questions may include images or Excel templates
        
        ### ✅ Guidelines
        - Answer each question completely and accurately
        - Explain your reasoning and approach
        - Use the clarification bot if you need help understanding a question
        - Watch the timer - manage your time effectively
        
        ### 🔧 Features Available
        - **Instant Q&A**: Ask for clarification without penalty
        - **Auto-save**: Your progress is automatically saved
        
        ### 📊 Evaluation Criteria
        You will be assessed on:
        - **Technical Accuracy** (Correct formulas, functions, procedures)
        - **Conceptual Understanding** (When and why to use specific features)
        - **Problem-solving Approach** (Logical thinking and methodology)
        - **Communication Skills** (Clear explanation of solutions)
        
        ### 🚨 Important Notes
        - Interview will **auto-submit** when timer reaches 00:00
        - All answers are saved automatically
        - You can skip questions if needed
        - Report will be available after completion
        """)
        
        # Add clarification bot in instructions
        show_clarification_bot()
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if st.button("🚀 Start Interview", use_container_width=True, type="primary"):
                try:
                    # Initialize interview manager with enhanced features
                    st.session_state.interview = InterviewManager(api_key=api_key if api_key.strip() else None,   # allow empty → fallback to .env
                                                                    model=model)
                    st.session_state.interview.set_candidate_info(
                        st.session_state.candidate_name,
                        st.session_state.candidate_email
                    )
                    st.session_state.interview_state = 'active'
                    st.session_state.chat_history = []
                    st.session_state.answer_key = 0
                    
                    # Add welcome message and first question
                    welcome_msg = st.session_state.interview.start_interview()
                    st.session_state.chat_history.append({"role": "assistant", "content": welcome_msg})
                    
                    # Get first question immediately
                    first_question = st.session_state.interview.get_next_question()
                    st.session_state.chat_history.append({"role": "assistant", "content": first_question})
                    
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error starting interview: {e}")
    
    # State: Active Interview
    elif st.session_state.interview_state == 'active':
        st.header("💬 Interview Session")
        
        # Progress indicator
        progress = st.session_state.interview.get_progress()
        st.progress(progress, text=f"Progress: {int(progress * 100)}% • Question {st.session_state.interview.current_question_index + 1}/8")
        
        # Chat interface
        chat_container = st.container()
        
        with chat_container:
            # Display chat history (clean, no scores visible)
            for message in st.session_state.chat_history:
                if message["role"] == "assistant":
                    with st.chat_message("assistant", avatar="🤖"):
                        # Filter out any scoring information
                        content = message["content"]
                        # Remove any lines containing scoring keywords
                        lines = content.split('\n')
                        clean_lines = []
                        for line in lines:
                            if not any(keyword in line.lower() for keyword in 
                                     ['score:', 'rubric:', 'accuracy:', 'total:', 'improvement:', 'rating:', 'points:']):
                                clean_lines.append(line)
                        clean_content = '\n'.join(clean_lines).strip()
                        st.markdown(clean_content)
                else:
                    with st.chat_message("user", avatar="👤"):
                        st.markdown(message["content"])
        
        # Clarification bot (always available during interview)
        show_clarification_bot()
        
        st.markdown("---")
        
        # Answer input section
        if not st.session_state.interview.is_interview_complete():
            # Auto-submit check
            if st.session_state.interview.is_time_up():
                st.session_state.interview.auto_submit_interview()
                st.session_state.interview_state = 'completed'
                st.error("⏰ Time expired! Interview has been automatically submitted.")
                st.rerun()
            
            # Answer input with dynamic key for clearing
            answer = st.text_area(
                "Your Answer:",
                height=120,
                placeholder="Type your detailed answer here... Include specific steps, formulas, or explanations.",
                key=f"answer_input_{st.session_state.answer_key}"
            )
            
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            
            with col2:
                if st.button("📤 Submit Answer", use_container_width=True, type="primary"):
                    if answer.strip():
                        # Add user message
                        st.session_state.chat_history.append({"role": "user", "content": answer})
                        
                        # Process answer
                        response = st.session_state.interview.process_answer(answer)
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
                        
                        # Clear answer box by incrementing key
                        st.session_state.answer_key += 1
                        
                        # Check completion
                        if st.session_state.interview.is_interview_complete():
                            st.session_state.interview_state = 'completed'
                        
                        st.rerun()
                    else:
                        st.error("❌ Please provide an answer before submitting.")
            
            with col3:
                if st.button("⏭️ Skip Question", use_container_width=True):
                    # Add skip message
                    st.session_state.chat_history.append({"role": "user", "content": "[Question Skipped]"})
                    
                    # Process skip
                    response = st.session_state.interview.skip_question()
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    
                    # Clear answer box
                    st.session_state.answer_key += 1
                    
                    # Check completion
                    if st.session_state.interview.is_interview_complete():
                        st.session_state.interview_state = 'completed'
                    
                    st.rerun()
            
            with col4:
                time_remaining = st.session_state.interview.get_remaining_time()
                minutes_left = time_remaining // 60
                if minutes_left <= 5:
                    st.error(f"⏰ {minutes_left}min left!")
                elif minutes_left <= 10:
                    st.warning(f"⏰ {minutes_left}min left")
                else:
                    st.info(f"⏰ {minutes_left}min left")
        else:
            st.success("🎉 Interview completed! Check the sidebar for your report.")
    
    # State: Completed
    elif st.session_state.interview_state == 'completed':
        st.header("🎉 Interview Completed!")
        
        if st.session_state.interview:
            # Show completion summary
            session_data = st.session_state.interview.get_session_data()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Questions Answered", f"{session_data['questions_answered']}/{session_data['total_questions']}")
            
            with col2:
                st.metric("Categories Covered", len(session_data['difficulty_breakdown']))
            
            with col3:
                st.metric("Time Taken", session_data.get('session_duration', 'Unknown'))
            
            st.markdown("---")
            
            st.success(f"""
            **Thank you, {st.session_state.candidate_name}!**
            
            Your Excel skills assessment has been completed and recorded. 
            
            **Next Steps:**
            - Your responses have been evaluated using advanced AI analysis
            - A detailed performance report has been generated
            - Results will be shared with relevant stakeholders
            - You may download your report using the sidebar
            
            **What was evaluated:**
            - Technical accuracy of your answers
            - Conceptual understanding of Excel features
            - Problem-solving approach and methodology
            - Communication clarity and completeness
            
            Thank you for your time and effort! 🚀
            """)
        
        # Reset option
        if st.button("🔄 Take Another Interview", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key not in ['admin_mode']:
                    del st.session_state[key]
            st.rerun()

if __name__ == "__main__":
    main()

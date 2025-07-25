import pandas as pd
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from email_validator import validate_email, EmailNotValidError
import os
import re
import time
import datetime
from pathlib import Path
import json
from dotenv import load_dotenv
load_dotenv()

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_KEY    = os.getenv("GROQ_API_KEY")
OPENAI_KEY  = os.getenv("OPENAI_API_KEY")


# File paths
QUESTIONS_PATH = "data/questions.csv"
RUBRIC_PATH = "eval/rubric.xlsx"
REPORTS_PATH = "reports/"

class InterviewManager:
    """Enhanced Interview Manager with Open-Source Embeddings and Advanced Features"""
    
    def __init__(self, api_key: str | None = None, model: str | None = None):
        # If user provided key in UI, use that; else fallback to env
        self.api_key =  OPENAI_KEY or GROQ_KEY
        if not self.api_key:
            raise ValueError("No API key provided. Set OPENAI_API_KEY or GROQ_API_KEY, or enter your OpenAI key in the UI.")
        # Decide base_url and model
        if api_key and api_key.strip():
            # User OpenAI key → use standard API and chosen model
            self.base_url = None
            self.model    = model or "gpt-4o-mini"
        else:
            # Fallback to GROQ
            self.base_url = GROQ_BASE_URL
            self.model    = model or "meta-llama/llama-4-scout-17b-16e-instruct"
        # Instantiate client once
        client_args = {"api_key": self.api_key}
        if self.base_url:
            client_args["base_url"] = self.base_url
        self.client = OpenAI(**client_args)
        
        # Initialize local embedding model
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Initialize components
        self._load_questions()
        self._setup_local_embeddings()
        
        # Session state
        self.candidate_name = ""
        self.candidate_email = ""
        self.current_question_index = 0
        self.questions_asked = []
        self.candidate_answers = []
        self.scores = []  # Internal scoring - not shown to candidate
        self.interview_complete = False
        self.session_start_time = None
        
        # Timer configuration (30 minutes)
        self.interview_duration = 30 * 60  # 30 minutes in seconds
        self.time_remaining = self.interview_duration
        self.timer_started = False
        
        # Interview configuration
        self.max_questions = 8
        self.difficulty_progression = ["Basic", "Basic", "Intermediate", "Intermediate", 
                                     "Intermediate", "Advanced", "Advanced", "Advanced"]
        
        # Clarification bot
        self.clarification_history = []
        
        # Create reports directory
        os.makedirs(REPORTS_PATH, exist_ok=True)
        
    def _load_questions(self):
        """Load questions from CSV file with image and Excel support"""
        try:
            self.questions_df = pd.read_csv(QUESTIONS_PATH)
            print(f"Loaded {len(self.questions_df)} questions successfully")
            
            # Ensure required columns exist
            required_columns = ['id', 'difficulty', 'category', 'question', 'ideal_answer']
            for col in required_columns:
                if col not in self.questions_df.columns:
                    raise ValueError(f"Missing required column: {col}")
                    
        except Exception as e:
            print(f"Error loading questions: {e}")
            # Fallback questions with enhanced structure
            self.questions_df = pd.DataFrame({
                'id': ['QEX1', 'QEX2', 'QEX3', 'QEX4', 'QEX5'],
                'difficulty': ['Basic', 'Intermediate', 'Advanced', 'Intermediate', 'Basic'],
                'category': ['Formulas', 'Data Analysis', 'Automation', 'Charts', 'Functions'],
                'question': [
                    'What does the SUM function do in Excel and provide its syntax?',
                    'How would you create a pivot table from a dataset?',
                    'Explain how to write a simple VBA macro for data automation.',
                    'How do you create a dynamic chart that updates automatically?',
                    'What is the difference between VLOOKUP and INDEX-MATCH?'
                ],
                'ideal_answer': [
                    'SUM function adds numerical values. Syntax: =SUM(number1, number2, ...) or =SUM(range)',
                    'Insert > PivotTable, select data range, drag fields to rows/columns/values areas, configure summary',
                    'Alt+F11 to open VBA Editor, Insert > Module, write Sub procedure with automation commands, save as macro-enabled',
                    'Create chart from data table, use dynamic named ranges or Table format for auto-updating',
                    'VLOOKUP searches first column and returns value from specified column. INDEX-MATCH is more flexible, can search any column'
                ],
                'question_type': ['text', 'text', 'text', 'text', 'text'],
                'image_path': ['', '', '', '', ''],
                'excel_template': ['', '', '', '', ''],
                'time_limit': [300, 600, 900, 600, 400]
            })
    
    def _setup_local_embeddings(self):
        """Set up local embeddings using SentenceTransformers and FAISS"""
        try:
            # Prepare text for embedding
            question_texts = []
            for _, row in self.questions_df.iterrows():
                text = f"{row['question']} {row['category']} {row['difficulty']}"
                question_texts.append(text)
            
            # Generate embeddings
            print("Generating local embeddings...")
            self.embeddings = self.embedding_model.encode(question_texts, convert_to_numpy=True)
            
            # Create FAISS index
            dimension = self.embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatL2(dimension)
            self.faiss_index.add(self.embeddings)
            
            print(f"Local embeddings initialized: {len(question_texts)} questions indexed")
            
        except Exception as e:
            print(f"Error setting up local embeddings: {e}")
            self.embeddings = None
            self.faiss_index = None
    
    def set_candidate_info(self, name, email):
        """Set candidate information with email validation"""
        try:
            # Validate email format
            validated_email = validate_email(email)
            self.candidate_email = validated_email.email
            self.candidate_name = name.strip()
            return True
        except EmailNotValidError as e:
            raise ValueError(f"Invalid email address: {str(e)}")
    
    def start_interview(self):
        """Start the interview with timer"""
        self.session_start_time = datetime.datetime.now()
        self.timer_started = True
        self.time_remaining = self.interview_duration
        
        welcome_message = f"""Hello {self.candidate_name}! 👋

I'm your AI Excel Mock Interviewer. I'll be assessing your Excel skills through a series of questions that will progressively increase in difficulty.

**Interview Details:**
- Duration: **30 minutes** ⏰
- Questions: {self.max_questions} total
- Difficulty: Basic → Intermediate → Advanced
- Auto-submit when timer expires

**Navigation:**
- Questions appear automatically
- Answer thoroughly and explain your reasoning
- Use the clarification bot if you need help understanding a question
- Timer is visible in the sidebar

Let's begin! 🚀"""
        
        return welcome_message
    
    def get_remaining_time(self):
        """Get remaining time in seconds"""
        if not self.timer_started or not self.session_start_time:
            return self.interview_duration
        
        elapsed = (datetime.datetime.now() - self.session_start_time).total_seconds()
        remaining = max(0, self.interview_duration - elapsed)
        return int(remaining)
    
    def is_time_up(self):
        """Check if time is up"""
        return self.get_remaining_time() <= 0
    
    def get_next_question(self):
        """Get the next question using local embeddings"""
        if self.current_question_index >= self.max_questions or self.is_time_up():
            self.interview_complete = True
            return self._generate_completion_message()
        
        # Determine target difficulty
        target_difficulty = self.difficulty_progression[self.current_question_index]
        
        # Filter questions by difficulty and not already asked
        available_questions = self.questions_df[
            (self.questions_df['difficulty'] == target_difficulty) &
            (~self.questions_df['id'].isin([q['id'] for q in self.questions_asked]))
        ]
        
        if available_questions.empty:
            # Fallback to any available question
            available_questions = self.questions_df[
                ~self.questions_df['id'].isin([q['id'] for q in self.questions_asked])
            ]
        
        if available_questions.empty:
            self.interview_complete = True
            return self._generate_completion_message()
        
        # Select question (prioritize by semantic similarity if previous answers available)
        if self.candidate_answers and self.faiss_index is not None:
            # Use last answer to find semantically related next question
            last_answer = self.candidate_answers[-1]['candidate_answer']
            if last_answer != "[SKIPPED]":
                query_embedding = self.embedding_model.encode([last_answer])
                available_indices = available_questions.index.tolist()
                available_embeddings = self.embeddings[available_indices]
                
                # Find most relevant question
                similarities = np.dot(query_embedding, available_embeddings.T).flatten()
                best_idx = available_indices[np.argmax(similarities)]
                selected_question = self.questions_df.loc[best_idx]
            else:
                selected_question = available_questions.sample(1).iloc[0]
        else:
            selected_question = available_questions.sample(1).iloc[0]
        
        # Store question details
        question_data = {
            'id': selected_question['id'],
            'difficulty': selected_question['difficulty'],
            'category': selected_question['category'],
            'question': selected_question['question'],
            'ideal_answer': selected_question['ideal_answer'],
            'question_type': selected_question.get('question_type', 'text'),
            'image_path': selected_question.get('image_path', ''),
            'excel_template': selected_question.get('excel_template', ''),
            'time_limit': selected_question.get('time_limit', 300),
            'index': self.current_question_index + 1
        }
        self.questions_asked.append(question_data)
        
        # Format question for display
        question_text = f"""**Question {self.current_question_index + 1} of {self.max_questions}**
📊 **Category:** {selected_question['category']}
🎯 **Level:** {selected_question['difficulty']}
⏱️ **Suggested Time:** {selected_question.get('time_limit', 300)//60} minutes

**{selected_question['question']}**

Please provide your answer below. Include specific steps or formulas where applicable."""
        
        return question_text
    
    def process_answer(self, candidate_answer):
        """Process candidate's answer with local evaluation"""
        if not self.questions_asked:
            return "Please wait for a question to be asked first."
        
        if self.is_time_up():
            self.interview_complete = True
            return "⏰ Time's up! Your interview has been automatically submitted."
        
        current_question = self.questions_asked[-1]
        
        # Store the answer
        self.candidate_answers.append({
            'question_id': current_question['id'],
            'question': current_question['question'],
            'candidate_answer': candidate_answer,
            'ideal_answer': current_question['ideal_answer'],
            'difficulty': current_question['difficulty'],
            'category': current_question['category'],
            'timestamp': datetime.datetime.now().isoformat()
        })
        
        # Internal scoring (not shown to candidate)
        score = self._evaluate_answer(candidate_answer, current_question['ideal_answer'])
        self.scores.append(score)
        
        # Generate feedback without revealing scores
        feedback = self._generate_feedback_without_scores(candidate_answer, current_question)
        
        self.current_question_index += 1
        
        # Get next question or complete interview
        if self.current_question_index >= self.max_questions or self.is_time_up():
            self.interview_complete = True
            return feedback + "\n\n" + self._generate_completion_message()
        else:
            next_question = self.get_next_question()
            return feedback + "\n\n---\n\n" + next_question
    
    def _evaluate_answer(self, candidate_answer, ideal_answer):
        """Enhanced evaluation using both local similarity and LLM scoring"""
        try:
            # Local similarity scoring using sentence embeddings
            candidate_embedding = self.embedding_model.encode([candidate_answer])
            ideal_embedding = self.embedding_model.encode([ideal_answer])
            
            similarity_score = np.dot(candidate_embedding, ideal_embedding.T)[0][0]
            
            # LLM evaluation if API key is available
            if self.client:
                evaluation_prompt = f"""
                As an Excel expert, evaluate this candidate's answer on a scale of 1-5:

                Ideal Answer: "{ideal_answer}"
                Candidate Answer: "{candidate_answer}"
                Semantic Similarity Score: {similarity_score:.3f}

                Scoring Criteria:
                - 5: Excellent - Complete, accurate, demonstrates deep understanding
                - 4: Good - Mostly correct with minor gaps
                - 3: Satisfactory - Basic understanding, some errors
                - 2: Poor - Limited understanding, significant errors
                - 1: Very Poor - Incorrect or meaningless

                Return JSON: {{"score": <1-5>, "accuracy": <0.0-1.0>, "reasoning": "<explanation>", "strengths": ["<list>"], "improvements": ["<list>"]}}
                """
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": evaluation_prompt}],
                    temperature=0.3,
                    response_format={"type": "json_object"}  # 💡 forces valid JSON
                )
                
                result = json.loads(response.choices[0].message.content)
                result['similarity_score'] = similarity_score
                return result

                
            else:
                # Fallback to similarity-based scoring
                score = max(1, min(5, int(similarity_score * 5) + 1))
                return {
                    "score": score,
                    "accuracy": similarity_score,
                    "reasoning": f"Similarity-based evaluation: {similarity_score:.3f}",
                    "strengths": ["Response provided"],
                    "improvements": ["Consider more detail" if similarity_score < 0.7 else "Good response"],
                    "similarity_score": similarity_score
                }
                
        except Exception as e:
            print(f"Error in evaluation: {e}")
            return {
                "score": 3,
                "accuracy": 0.6,
                "reasoning": "Unable to evaluate automatically",
                "strengths": ["Response provided"],
                "improvements": ["More detail needed"],
                "similarity_score": 0.5
            }
    
    def _generate_feedback_without_scores(self, candidate_answer, question_data):
        """Generate encouraging feedback without revealing scores"""
        try:
            if self.client:
                feedback_prompt = f"""
                    You a supportive Excel interviewer. Provide exactly two sentences of encouraging feedback that:
                    - Do NOT mention scores, numbers, or evaluative language.
                    - Do NOT pose any questions.
                    - Acknowledge the candidate effort and smooth the transition to the next question.

                    Candidates answer: "{candidate_answer}"

                    Respond with two positive, statement only sentences.
                    """
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": feedback_prompt}],
                    temperature=0.7
                )
                
                return response.choices[0].message.content.strip()
            else:
                return "Thank you for your detailed answer. Let's continue with the next question."
                
        except Exception as e:
            print(f"Error generating feedback: {e}")
            return "Thank you for your answer. Moving on to the next question."
    
    def ask_clarification(self, user_question: str) -> str:
        """
        Respond to candidate requests without leaking answers.
        • Returns a short, safe reply for allowed topics
        • Warns the candidate if they request help on solving a test question
        • Escalates genuine tech/support issues to the admin log
        """
        # --- Basic filtering -------------------------------------------------
        q_lower = user_question.lower().strip()

        # Any attempt to get the answer or hints?
        if any(kw in q_lower for kw in
               ["what's the answer", "how do i solve", "give me formula",
                "tell me the answer", "vlookup solution", "macro code"]):
            warn_msg = ("⚠️ Please note: asking for solutions to assessment "
                        "questions violates test policy and may result in "
                        "disqualification.")
            # Log but do not reveal anything
            self.clarification_history.append({
                "question": user_question,
                "response": warn_msg,
                "flag": "policy_warning",
                "timestamp": datetime.datetime.now().isoformat()
            })
            return warn_msg

        # Tech‐support / environment issues?
        tech_keywords = ["page not loading", "button doesn't work", "camera",
                         "microphone", "audio", "video", "error", "bug",
                         "freeze", "crash", "internet"]
        if any(kw in q_lower for kw in tech_keywords):
            admin_note = (f"🚨 TECH ISSUE from {self.candidate_name} "
                          f"({self.candidate_email}): {user_question}")
            # Store a high-priority flag so admins can act
            self.clarification_history.append({
                "question": user_question,
                "response": "Admin notified",
                "flag": "tech_issue",
                "timestamp": datetime.datetime.now().isoformat(),
                "admin_note": admin_note
            })
            ALERTS_LOG = Path("data/alerts.json")
            # inside the tech‐support branch, before return:
            alert_entry = {
                "candidate_name": self.candidate_name,
                "candidate_email": self.candidate_email,
                "question": user_question,
                "timestamp": datetime.datetime.now().strftime("%d %b %Y, %H:%M:%S")
            }
            # append to shared file
            with ALERTS_LOG.open("a", encoding="utf-8") as f:
                f.write(json.dumps(alert_entry) + "\n")
            return ("It looks like you are experiencing a technical issue. "
                    "A supervisor has been notified and will assist you shortly.")

        # Allowed “house-keeping” topics (duration, navigation, rules, etc.)
        allowed_prompt = (
            "You are the Excel Mock-Interview assistant. "
            "Answer ONLY questions about test logistics, rules, duration, "
            "navigation, or permitted actions. DO NOT provide any hints or "
            "content that would help solve a test question."
            "NEVER suggest the candidate refresh or reload the page or "
            "perform any browser actions to resolve issues.\n\n"
            f"Candidate: {user_question}\n\n"
            "Assistant (2 sentences max):"
        )

        try:
            if self.client:
                reply = (
                    self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": allowed_prompt}],
                        temperature=0.3,
                        response_format={"type": "text"}
                    ).choices[0].message.content.strip()
                )
            else:
                # Fallback canned reply (no API key)
                reply = ("This assessment lasts 30 minutes and consists of "
                         "8 questions. Use the sidebar to monitor time and "
                         "the buttons below each prompt to navigate.")
        except Exception as e:
            print("Clarification LLM error:", e)
            reply = ("I amsorry, I am currently unable to process that request. "
                     "Please continue with the assessment or contact support.")

        # Log safe clarification
        self.clarification_history.append({
            "question": user_question,
            "response": reply,
            "flag": "normal",
            "timestamp": datetime.datetime.now().isoformat()
        })
        return reply

    
    def skip_question(self):
        """Handle question skipping"""
        if not self.questions_asked:
            return "No question to skip."
        
        current_question = self.questions_asked[-1]
        
        # Record skip
        self.candidate_answers.append({
            'question_id': current_question['id'],
            'question': current_question['question'],
            'candidate_answer': "[SKIPPED]",
            'ideal_answer': current_question['ideal_answer'],
            'difficulty': current_question['difficulty'],
            'category': current_question['category'],
            'timestamp': datetime.datetime.now().isoformat()
        })
        
        # Internal scoring for skip
        skip_score = {
            "score": 1,
            "accuracy": 0.0,
            "reasoning": "Question skipped by candidate",
            "strengths": [],
            "improvements": ["Consider attempting the question"],
            "similarity_score": 0.0
        }
        self.scores.append(skip_score)
        
        self.current_question_index += 1
        
        # Get next question or complete
        if self.current_question_index >= self.max_questions or self.is_time_up():
            self.interview_complete = True
            return "Question skipped. " + self._generate_completion_message()
        else:
            next_question = self.get_next_question()
            return "Question skipped. Let's move on.\n\n---\n\n" + next_question
    
    def _generate_completion_message(self):
        """Generate interview completion message"""
        time_taken = "Full duration" if self.is_time_up() else f"{(datetime.datetime.now() - self.session_start_time).total_seconds()//60:.0f} minutes"
        
        return f"""🎉 **Interview Complete!**

Thank you, {self.candidate_name}, for completing the Excel skills assessment!

**Session Summary:**
- Time taken: {time_taken}
- Questions answered: {len([a for a in self.candidate_answers if a['candidate_answer'] != '[SKIPPED]'])}/{len(self.candidate_answers)}
- Topics covered: {len(set(a['category'] for a in self.candidate_answers))} categories

**What happens next:**
- Your detailed performance report is being generated
- An admin will review your responses and provide feedback
- Results will be shared via email within 24 hours

Great job completing the interview! 👏"""
    
    def auto_submit_interview(self):
        """Auto-submit interview when time is up"""
        if not self.interview_complete:
            self.interview_complete = True
            
            # If there's an active question, record it as timed out
            if self.questions_asked and len(self.candidate_answers) < len(self.questions_asked):
                current_question = self.questions_asked[-1]
                self.candidate_answers.append({
                    'question_id': current_question['id'],
                    'question': current_question['question'],
                    'candidate_answer': "[TIME_EXPIRED]",
                    'ideal_answer': current_question['ideal_answer'],
                    'difficulty': current_question['difficulty'],
                    'category': current_question['category'],
                    'timestamp': datetime.datetime.now().isoformat()
                })
                
                timeout_score = {
                    "score": 1,
                    "accuracy": 0.0,
                    "reasoning": "Time expired before answer submitted",
                    "strengths": [],
                    "improvements": ["Time management"],
                    "similarity_score": 0.0
                }
                self.scores.append(timeout_score)
    
    def is_interview_complete(self):
        """Check if interview is complete"""
        return self.interview_complete or self.is_time_up()
    
    def get_progress(self):
        """Get interview progress as percentage"""
        return min(self.current_question_index / self.max_questions, 1.0)
    
    def get_session_data(self):
        """Get complete session data for report generation"""
        session_data = {
            'candidate_name': self.candidate_name,
            'candidate_email': self.candidate_email,
            'session_date': self.session_start_time.strftime("%Y-%m-%d %H:%M:%S") if self.session_start_time else "Unknown",
            'session_duration': f"{(datetime.datetime.now() - self.session_start_time).total_seconds()//60:.0f} minutes" if self.session_start_time else "Unknown",
            'questions_and_answers': self.candidate_answers,
            'scores': self.scores,
            'clarifications': self.clarification_history,
            'total_questions': len(self.candidate_answers),
            'questions_answered': len([a for a in self.candidate_answers if a['candidate_answer'] not in ['[SKIPPED]', '[TIME_EXPIRED]']]),
            'questions_skipped': len([a for a in self.candidate_answers if a['candidate_answer'] == '[SKIPPED]']),
            'timed_out_questions': len([a for a in self.candidate_answers if a['candidate_answer'] == '[TIME_EXPIRED]']),
            'average_score': np.mean([s['score'] for s in self.scores]) if self.scores else 0,
            'average_similarity': np.mean([s.get('similarity_score', 0) for s in self.scores]) if self.scores else 0,
            'difficulty_breakdown': {
                'Basic': len([a for a in self.candidate_answers if a['difficulty'] == 'Basic']),
                'Intermediate': len([a for a in self.candidate_answers if a['difficulty'] == 'Intermediate']),
                'Advanced': len([a for a in self.candidate_answers if a['difficulty'] == 'Advanced'])
            },
            'category_performance': self._get_category_performance(),
            'completion_status': 'Completed' if self.interview_complete else 'In Progress'
        }
        
        return session_data
    
    def _get_category_performance(self):
        """Get performance breakdown by category"""
        if not self.scores or not self.candidate_answers:
            return {}
        
        category_scores = {}
        for answer, score in zip(self.candidate_answers, self.scores):
            category = answer['category']
            if category not in category_scores:
                category_scores[category] = []
            category_scores[category].append(score['score'])
        
        return {cat: np.mean(scores) for cat, scores in category_scores.items()}

# interview/reports.py
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from datetime import datetime
import os
import io

def generate_pdf_report(session_data):
    """Generate PDF report from session data with proper field mapping"""
    
    # Extract info from session_data with correct field names
    candidate_name = session_data.get('candidate_name', 'Unknown')
    candidate_email = session_data.get('candidate_email', 'N/A')
    questions_and_answers = session_data.get('questions_and_answers', [])
    scores = session_data.get('scores', [])
    overall_score = session_data.get('average_score', 0)
    session_date = session_data.get('session_date', 'Unknown')
    session_duration = session_data.get('session_duration', 'Unknown')
    
    # Create filename and ensure reports directory exists
    filename = f"interview_report_{candidate_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    filepath = os.path.join("reports", filename)
    os.makedirs("reports", exist_ok=True)
    
    # Create PDF document
    doc = SimpleDocTemplate(filepath, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title = Paragraph(f"Excel Skills Assessment Report", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 0.2*inch))
    
    # Candidate Information
    candidate_info = f"""
    <b>Candidate:</b> {candidate_name}<br/>
    <b>Email:</b> {candidate_email}<br/>
    <b>Assessment Date:</b> {session_date}<br/>
    <b>Duration:</b> {session_duration}<br/>
    <b>Overall Score:</b> {overall_score:.1f}/5.0
    """
    story.append(Paragraph(candidate_info, styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    # Summary Statistics
    total_questions = len(questions_and_answers)
    answered_questions = len([q for q in questions_and_answers if q.get('candidate_answer', '') not in ['[SKIPPED]', '[TIME_EXPIRED]', '']])
    skipped_questions = len([q for q in questions_and_answers if q.get('candidate_answer', '') == '[SKIPPED]'])
    timed_out_questions = len([q for q in questions_and_answers if q.get('candidate_answer', '') == '[TIME_EXPIRED]'])
    
    summary_data = [
        ['Metric', 'Count'],
        ['Total Questions', str(total_questions)],
        ['Questions Answered', str(answered_questions)],
        ['Questions Skipped', str(skipped_questions)],
        ['Timed Out Questions', str(timed_out_questions)]
    ]
    
    summary_table = Table(summary_data)
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(summary_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Individual Question Analysis
    story.append(Paragraph("Detailed Question Analysis", styles['Heading1']))
    story.append(Spacer(1, 0.2*inch))
    
    for i, qa in enumerate(questions_and_answers, 1):
        # Extract data with correct field names
        question = qa.get('question', 'N/A')
        # THIS IS THE KEY FIX - use 'candidate_answer' instead of 'answer'
        answer = qa.get('candidate_answer', 'N/A')
        ideal_answer = qa.get('ideal_answer', 'N/A')
        difficulty = qa.get('difficulty', 'N/A')
        category = qa.get('category', 'N/A')
        
        # Get scoring data if available
        score_data = scores[i-1] if i-1 < len(scores) else {}
        score = score_data.get('score', 'N/A')
        reasoning = score_data.get('reasoning', 'No detailed feedback available')
        
        # Generate feedback based on score or use stored feedback
        if 'feedback' in qa and qa['feedback']:
            feedback = qa['feedback']
        else:
            # Generate basic feedback from scoring data
            if isinstance(score, (int, float)):
                if score >= 4:
                    feedback = f"Excellent response! {reasoning}"
                elif score >= 3:
                    feedback = f"Good understanding shown. {reasoning}"
                elif score >= 2:
                    feedback = f"Basic understanding demonstrated. {reasoning}"
                else:
                    feedback = f"Consider reviewing this topic. {reasoning}"
            else:
                feedback = reasoning
        
        # Add question section
        question_title = f"Question {i}: {category} ({difficulty})"
        story.append(Paragraph(question_title, styles['Heading2']))
        
        question_text = f"<b>Q:</b> {question}"
        story.append(Paragraph(question_text, styles['Normal']))
        story.append(Spacer(1, 0.1*inch))
        
        answer_text = f"<b>Candidate Answer:</b> {answer}"
        story.append(Paragraph(answer_text, styles['Normal']))
        story.append(Spacer(1, 0.1*inch))
        
        if score != 'N/A':
            score_text = f"<b>Score:</b> {score}/5"
            story.append(Paragraph(score_text, styles['Normal']))
            story.append(Spacer(1, 0.1*inch))
        
        feedback_text = f"<b>Feedback:</b> {feedback}"
        story.append(Paragraph(feedback_text, styles['Normal']))
        story.append(Spacer(1, 0.1*inch))
        
        # Add ideal answer for reference
        ideal_text = f"<b>Model Answer:</b> {ideal_answer}"
        story.append(Paragraph(ideal_text, styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
    
    # Performance Summary
    if scores:
        avg_score = sum([s.get('score', 0) for s in scores]) / len(scores)
        performance_text = f"""
        <b>Performance Summary:</b><br/>
        Average Score: {avg_score:.2f}/5.0<br/>
        Performance Level: {'Excellent' if avg_score >= 4 else 'Good' if avg_score >= 3 else 'Satisfactory' if avg_score >= 2 else 'Needs Improvement'}
        """
        story.append(Paragraph(performance_text, styles['Normal']))
    
    # Build PDF
    doc.build(story)
    
    # Return the PDF as bytes for download
    with open(filepath, 'rb') as f:
        pdf_data = f.read()
    
    return pdf_data
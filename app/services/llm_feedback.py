"""
LLM feedback service for TalentMatch AI.

This module generates recruiter-style feedback using Gemini/OpenAI
when API keys are available, with a rule-based fallback.
"""

from app.config import (
    DEFAULT_LLM_PROVIDER,
    GEMINI_API_KEY,
    OPENAI_API_KEY,
)


class LLMFeedbackGenerator:
    """
    Generates human-readable feedback for candidates and recruiters.
    """

    def generate_feedback(
        self,
        overall_score: float,
        matched_skills: list[str],
        missing_skills: list[str],
        skill_score: float | None = None,
        ats_score: float | None = None,
        experience_score: float | None = None,
    ) -> dict:
        """
        Generate recommendations and recruiter summary.
        """

        prompt = self._build_prompt(
            overall_score,
            matched_skills,
            missing_skills,
        )

        if DEFAULT_LLM_PROVIDER == "gemini" and GEMINI_API_KEY:
            return self._generate_with_gemini(
                prompt,
                overall_score,
                matched_skills,
                missing_skills,
            )

        if DEFAULT_LLM_PROVIDER == "openai" and OPENAI_API_KEY:
            return self._generate_with_openai(
                prompt,
                overall_score,
                matched_skills,
                missing_skills,
            )

        return self._rule_based_feedback(
            overall_score,
            matched_skills,
            missing_skills,
            skill_score=skill_score,
            ats_score=ats_score,
            experience_score=experience_score,
        )

    def _build_prompt(
        self,
        overall_score: float,
        matched_skills: list[str],
        missing_skills: list[str],
    ) -> str:
        """
        Build recruiter-style prompt.
        """

        return f"""
You are an experienced technical recruiter.

Analyze this candidate-job match.

Overall match score: {overall_score}%

Matched skills:
{", ".join(matched_skills) if matched_skills else "None"}

Missing skills:
{", ".join(missing_skills) if missing_skills else "None"}

        Return concise output with:
1. Candidate Recommendation
2. Key Strengths
3. Major Skill Gaps
4. Suggested Next Steps
5. Interview Recommendation

Keep the tone professional and realistic.
"""

    def _generate_with_gemini(
        self,
        prompt: str,
        overall_score: float,
        matched_skills: list[str],
        missing_skills: list[str],
    ) -> dict:
        """
        Generate feedback using Gemini API.
        """

        try:
            import google.generativeai as genai

            genai.configure(api_key=GEMINI_API_KEY)

            model = genai.GenerativeModel("gemini-1.5-flash")

            response = model.generate_content(prompt)

            text = response.text.strip()

            return {
                "recommendations": [text],
                "recruiter_summary": text,
                "structured_feedback": self._structured_rule_based_feedback(
                    overall_score,
                    matched_skills,
                    missing_skills,
                ),
            }

        except Exception:
            return self._rule_based_feedback(
                overall_score,
                matched_skills,
                missing_skills,
            )

    def _generate_with_openai(
        self,
        prompt: str,
        overall_score: float,
        matched_skills: list[str],
        missing_skills: list[str],
    ) -> dict:
        """
        Generate feedback using OpenAI API.
        """

        try:
            from openai import OpenAI

            client = OpenAI(api_key=OPENAI_API_KEY)

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an experienced technical recruiter.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0.3,
            )

            text = response.choices[0].message.content.strip()

            return {
                "recommendations": [text],
                "recruiter_summary": text,
                "structured_feedback": self._structured_rule_based_feedback(
                    overall_score,
                    matched_skills,
                    missing_skills,
                ),
            }

        except Exception:
            return self._rule_based_feedback(
                overall_score,
                matched_skills,
                missing_skills,
            )

    def _rule_based_feedback(
        self,
        overall_score: float,
        matched_skills: list[str],
        missing_skills: list[str],
        skill_score: float | None = None,
        ats_score: float | None = None,
        experience_score: float | None = None,
    ) -> dict:
        """
        Fallback feedback when no LLM API key is available.
        """

        structured_feedback = self._structured_rule_based_feedback(
            overall_score,
            matched_skills,
            missing_skills,
            skill_score=skill_score,
            ats_score=ats_score,
            experience_score=experience_score,
        )

        recommendations = structured_feedback["suggested_next_steps"]

        if missing_skills:
            recommendations.append(
                "Improve or add evidence for the missing skills: "
                + ", ".join(missing_skills[:8])
            )

        if matched_skills:
            recommendations.append(
                "Keep the matched skills clearly visible in the resume: "
                + ", ".join(matched_skills[:8])
            )

        recruiter_summary = structured_feedback["candidate_recommendation"]

        if not recommendations:
            recommendations.append(
                "Add more role-specific skills, measurable project outcomes, "
                "and relevant tools to improve resume alignment."
            )

        return {
            "recommendations": recommendations,
            "recruiter_summary": recruiter_summary,
            "structured_feedback": structured_feedback,
        }

    @staticmethod
    def _structured_rule_based_feedback(
        overall_score: float,
        matched_skills: list[str],
        missing_skills: list[str],
        skill_score: float | None = None,
        ats_score: float | None = None,
        experience_score: float | None = None,
    ) -> dict:
        """
        Structured fallback that works without paid API keys.
        """

        skill_score = skill_score if skill_score is not None else overall_score
        ats_score = ats_score if ats_score is not None else overall_score
        experience_score = experience_score if experience_score is not None else overall_score
        critical_missing = {
            skill
            for skill in missing_skills
            if skill.lower() in {"python", "machine learning", "fastapi", "docker", "sql"}
        }

        if overall_score >= 80 and skill_score >= 70 and ats_score >= 70:
            recommendation = (
                "Strong candidate-job fit. Prioritize this candidate for recruiter "
                "review and validate project depth during screening."
            )
            interview_recommendation = "Strong Shortlist"
        elif overall_score >= 65 and skill_score >= 55:
            recommendation = (
                "Moderate candidate-job fit. The candidate has relevant signals, "
                "but gaps should be checked before shortlisting."
            )
            interview_recommendation = "Shortlist"
        elif overall_score >= 60 and skill_score >= 45:
            recommendation = (
                "Promising junior-level fit. Consider a conditional shortlist if "
                "the role can support mentoring and practical project validation."
            )
            interview_recommendation = "Junior-Level Fit"
        elif overall_score >= 45 or (skill_score >= 45 and ats_score >= 55):
            recommendation = (
                "Partial candidate-job fit. This profile may still be suitable for "
                "internship or junior roles if the candidate can demonstrate learning "
                "velocity and project ownership."
            )
            interview_recommendation = (
                "Needs More Evidence"
                if len(critical_missing) >= 3 or experience_score < 45
                else "Conditional Shortlist"
            )
        else:
            recommendation = (
                "Low candidate-job fit based on the available resume evidence. "
                "The candidate may still be relevant for internship or junior roles, "
                "but more evidence is needed before shortlist."
            )
            interview_recommendation = "Needs More Evidence"

        key_strengths = (
            [
                f"Evidence of {skill}"
                for skill in matched_skills[:6]
            ]
            if matched_skills
            else ["No direct skill matches were detected from the resume text."]
        )

        major_skill_gaps = (
            missing_skills[:8]
            if missing_skills
            else ["No major skill gaps were detected from the job description."]
        )

        suggested_next_steps = []

        if missing_skills:
            suggested_next_steps.append(
                "Ask focused screening questions around: "
                + ", ".join(missing_skills[:5])
            )

        if matched_skills:
            suggested_next_steps.append(
                "Validate practical experience with: "
                + ", ".join(matched_skills[:5])
            )

        suggested_next_steps.append(
            "Review project outcomes, seniority, and communication fit before final decision."
        )

        return {
            "candidate_recommendation": recommendation,
            "key_strengths": key_strengths,
            "major_skill_gaps": major_skill_gaps,
            "suggested_next_steps": suggested_next_steps,
            "interview_recommendation": interview_recommendation,
        }

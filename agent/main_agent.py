import asyncio
import re
from typing import Dict


class MainAgent:
    """
    RAG Agent mô phỏng với knowledge base nội bộ.
    V1: retrieval cơ bản (keyword match đơn giản)
    V2: retrieval nâng cao (keyword + phrase match, reranking)
    """

    KNOWLEDGE_BASE = {
        "doc_password_1":
        "Password Reset Procedure: Users can reset their password by navigating to "
        "Settings > Security > Reset Password. An email verification link will be sent "
        "to the registered email address. The link expires within 24 hours.",
        "doc_account_1":
        "Account Creation: New users can create an account by clicking 'Sign Up' on "
        "the login page. Required fields include email, username, and password. "
        "Email verification is required within 48 hours.",
        "doc_security_1":
        "Security Best Practices: Use strong passwords with at least 12 characters "
        "including numbers, symbols, and uppercase letters. Enable two-factor "
        "authentication (2FA) for additional security.",
        "doc_billing_1":
        "Subscription Plans: Basic ($9.99/month): 1 user, 10GB storage. "
        "Pro ($29.99/month): 5 users, 100GB storage. "
        "Enterprise ($99.99/month): Unlimited users, 1TB storage.",
        "doc_api_1":
        "API Integration Guide: REST API endpoints available at "
        "https://api.example.com/v1. Authentication via Bearer token. "
        "Rate limit: 1000 requests/hour for Pro plans.",
        "doc_data_1":
        "Data Privacy Policy: User data is encrypted at rest using AES-256. "
        "Data is processed in accordance with GDPR and CCPA regulations. "
        "Data retention period is 12 months after account deletion.",
        "doc_troubleshoot_1":
        "Common Troubleshooting: If the application is not responding, try clearing "
        "browser cache, disabling browser extensions, or switching to a supported "
        "browser (Chrome, Firefox, Edge).",
        "doc_install_1":
        "Installation Guide: System requirements: Windows 10+, macOS 12+, or Linux "
        "(Ubuntu 20.04+). Minimum 4GB RAM, 500MB disk space. Run the installer "
        "and follow the setup wizard.",
        "doc_config_1":
        "Configuration Options: Key configuration parameters include: "
        "max_session_timeout (default: 3600s), max_login_attempts (default: 5), "
        "enable_2fa (default: false), notification_email (default: admin@example.com).",
        "doc_perf_1":
        "Performance Optimization: Recommended settings include enabling caching, "
        "using a CDN for static assets, database query optimization, and implementing "
        "lazy loading for large datasets.",
    }

    # Keyword index cho mỗi document
    KEYWORD_INDEX = {
        "password": ["doc_password_1", "doc_security_1"],
        "reset": ["doc_password_1"],
        "account": ["doc_account_1", "doc_billing_1"],
        "create": ["doc_account_1"],
        "signup": ["doc_account_1"],
        "security": ["doc_security_1", "doc_data_1"],
        "2fa": ["doc_security_1"],
        "billing": ["doc_billing_1"],
        "subscription": ["doc_billing_1"],
        "api": ["doc_api_1"],
        "integration": ["doc_api_1"],
        "privacy": ["doc_data_1"],
        "data": ["doc_data_1", "doc_perf_1"],
        "gdpr": ["doc_data_1"],
        "troubleshoot": ["doc_troubleshoot_1"],
        "error": ["doc_troubleshoot_1"],
        "install": ["doc_install_1"],
        "setup": ["doc_install_1", "doc_config_1"],
        "config": ["doc_config_1"],
        "performance": ["doc_perf_1"],
        "optimize": ["doc_perf_1"],
        "cache": ["doc_perf_1"],
        "plan": ["doc_billing_1"],
        "price": ["doc_billing_1"],
        "rate": ["doc_api_1"],
        "limit": ["doc_api_1", "doc_config_1"],
    }

    def __init__(self, version: str = "V1"):
        self.name = f"SupportAgent-{version}"
        self.version = version
        # V1: model rẻ, V2: model tốt hơn
        self.model = "gpt-4o-mini" if version == "V1" else "gpt-4o"

    def _extract_keywords(self, question: str) -> list:
        """Trích xuất keywords từ câu hỏi."""
        stop_words = {
            "the", "a", "an", "in", "on", "at", "to", "for", "of", "with",
            "by", "is", "are", "was", "were", "be", "been", "being", "have",
            "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "i", "you", "he", "she", "it", "we",
            "they", "this", "that", "these", "those", "how", "what", "why",
            "when", "where", "which", "who", "whom", "làm", "thế", "nào",
            "cách", "gì", "tại", "sao", "ở", "đâu", "khi", "about", "can",
            "not", "or", "my", "me", "your", "please", "help",
        }
        words = re.findall(r'\b\w+\b', question.lower())
        return [w for w in words if w not in stop_words and len(w) > 2]

    def _retrieve_v1(self, question: str):
        """V1: keyword match cơ bản, không có xếp hạng thông minh."""
        keywords = self._extract_keywords(question)
        # Gộp tất cả doc match được
        matched_doc_ids = set()
        for kw in keywords:
            if kw in self.KEYWORD_INDEX:
                matched_doc_ids.update(self.KEYWORD_INDEX[kw])

        # Sắp xếp theo số lượng keyword match (giảm dần)
        scored = []
        for doc_id in matched_doc_ids:
            doc_lower = self.KNOWLEDGE_BASE[doc_id].lower()
            score = sum(1 for kw in keywords if kw in doc_lower)
            scored.append((doc_id, score))
        scored.sort(key=lambda x: x[1], reverse=True)

        retrieved_ids = [doc_id for doc_id, _ in scored]
        contexts = [self.KNOWLEDGE_BASE[doc_id] for doc_id in retrieved_ids]
        return contexts, retrieved_ids

    def _retrieve_v2(self, question: str):
        """V2: keyword + phrase match + reranking."""
        keywords = self._extract_keywords(question)
        words = keywords

        # Tạo bigrams
        bigrams = set()
        for i in range(len(words) - 1):
            bigrams.add(f"{words[i]} {words[i+1]}")

        matched_doc_ids = set()
        for kw in keywords:
            if kw in self.KEYWORD_INDEX:
                matched_doc_ids.update(self.KEYWORD_INDEX[kw])

        scored = []
        for doc_id in matched_doc_ids:
            doc_lower = self.KNOWLEDGE_BASE[doc_id].lower()
            keyword_score = sum(2 for kw in keywords if kw in doc_lower)
            phrase_score = sum(5 for phrase in bigrams if phrase in doc_lower)
            scored.append((doc_id, keyword_score + phrase_score))
        scored.sort(key=lambda x: x[1], reverse=True)

        retrieved_ids = [doc_id for doc_id, _ in scored]
        contexts = [self.KNOWLEDGE_BASE[doc_id] for doc_id in retrieved_ids]
        return contexts, retrieved_ids

    async def query(self, question: str) -> Dict:
        """Mô phỏng RAG pipeline: retrieval + generation."""
        await asyncio.sleep(0.3 if self.version == "V1" else 0.2)

        if self.version == "V1":
            contexts, retrieved_ids = self._retrieve_v1(question)
        else:
            contexts, retrieved_ids = self._retrieve_v2(question)

        # Mô phỏng generation dựa trên context
        if contexts:
            q_short = question[:60].lower()
            answer = f"Dựa trên tài liệu hệ thống, {q_short}... "
            snippets = [c[:80] for c in contexts[:2]]
            answer += " ".join(snippets)
        else:
            answer = "Xin lỗi, tôi không tìm thấy thông tin liên quan trong cơ sở dữ liệu."

        # Token tracking mô phỏng
        input_tokens = len(question.split()) * 2 + sum(len(c.split()) for c in contexts)
        output_tokens = len(answer.split()) * 2

        return {
            "answer": answer,
            "contexts": contexts,
            "retrieved_ids": retrieved_ids,
            "metadata": {
                "model": self.model,
                "version": self.version,
                "input_tokens": int(input_tokens),
                "output_tokens": int(output_tokens),
                "total_tokens": int(input_tokens + output_tokens),
            }
        }


if __name__ == "__main__":
    agent = MainAgent("V2")
    async def test():
        resp = await agent.query("How do I reset my password?")
        print(f"Retrieved IDs: {resp['retrieved_ids']}")
        print(f"Answer: {resp['answer'][:100]}")
    asyncio.run(test())

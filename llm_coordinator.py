# llm_coordinator.py
import os
import time
import base64
import requests
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from config import CONFIG

load_dotenv()
logger = logging.getLogger(__name__)

class GigaChatCoordinator:
    """
    Production-grade GigaChat integration for FINOVA PILOT.
    Handles OAuth2 token lifecycle, request retry, and structured prompt generation.
    """
    def __init__(self, config: FINOVAConfig = CONFIG):
        self.config = config
        self.client_id = os.getenv("GIGACHAT_CLIENT_ID")
        self.client_secret = os.getenv("GIGACHAT_CLIENT_SECRET")
        
        if not self.client_id or not self.client_secret:
            raise ValueError(
                "GIGACHAT_CLIENT_ID and GIGACHAT_CLIENT_SECRET must be defined in .env or environment variables."
            )

        self.auth_url = config.GIGACHAT_AUTH_URL
        self.api_url = config.GIGACHAT_API_URL
        self.scope = config.GIGACHAT_SCOPE

        self._access_token: Optional[str] = None
        self._token_expires_at: float = 0.0
        self._request_token()

    def _get_basic_auth_header(self) -> str:
        """Construct Basic Auth header from client_id:client_secret"""
        credentials = f"{self.client_id}:{self.client_secret}"
        return f"Basic {base64.b64encode(credentials.encode('utf-8')).decode('utf-8')}"

    def _request_token(self) -> None:
        """Obtain new access token from Sber GigaChat OAuth endpoint"""
        headers = {
            "Authorization": self._get_basic_auth_header(),
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json"
        }
        data = {"scope": self.scope}
        
        try:
            response = requests.post(self.auth_url, headers=headers, data=data, timeout=15)
            response.raise_for_status()
            token_data = response.json()
            
            self._access_token = token_data["access_token"]
            # expires_at comes in milliseconds
            self._token_expires_at = token_data.get("expires_at", 0) / 1000.0
            logger.info(f"GigaChat token obtained. Expires at: {datetime.fromtimestamp(self._token_expires_at)}")
        except requests.RequestException as e:
            logger.error(f"Failed to obtain GigaChat token: {e}")
            raise RuntimeError(f"GigaChat authentication failed: {e}")

    def _ensure_valid_token(self) -> str:
        """Return valid token, refresh if expiring within 60 seconds"""
        if not self._access_token or (self._token_expires_at - time.time()) < 60:
            self._request_token()
        return self._access_token

    def generate_advice(self, profile: str, allocation: Dict[str, float], feasibility: Dict[str, Any], market_regime: str) -> str:
        """
        Generate investment recommendation via GigaChat API.
        Args:
            profile: Behavioral profile name (e.g., 'panicker', 'degen')
            allocation: Dict with stock/bond/crypto allocations
            feasibility: FeasibilityGuard result dict
            market_regime: Current market regime string
        Returns:
            Formatted recommendation text
        """
        token = self._ensure_valid_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "RqUID": os.urandom(16).hex()  # UUID v4 format for request tracing
        }

        prompt = self._build_prompt(profile, allocation, feasibility, market_regime)
        payload = {
            "model": "GigaChat",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 1024
        }

        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            
            # Handle token expiration during API call
            if response.status_code == 401:
                token = self._ensure_valid_token()
                headers["Authorization"] = f"Bearer {token}"
                response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
                
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
            
        except requests.RequestException as e:
            logger.error(f"GigaChat API call failed: {e}")
            raise RuntimeError(f"LLM recommendation generation failed: {e}")

    def _build_prompt(self, profile: str, allocation: Dict[str, float], feasibility: Dict[str, Any], market_regime: str) -> str:
        """Construct structured prompt for GigaChat"""
        cagr = feasibility.get('required_cagr', 0.0)
        risk_level = feasibility.get('risk_level', 'green')
        stock = allocation.get('stock_allocation', 0.0)
        bond = allocation.get('bond_allocation', 0.0)
        crypto = allocation.get('crypto_allocation', 0.0)

        return (
            f"Вы — профессиональный инвестиционный советник FINOVA. "
            f"Профиль инвестора: {profile}. Текущий режим рынка: {market_regime}. "
            f"Требуемая доходность: {cagr:.1%} годовых (статус: {risk_level}). "
            f"Рекомендуемая аллокация: акции {stock:.0%}, облигации {bond:.0%}, крипто {crypto:.0%}. "
            f"Сформируйте краткую рекомендацию (до 3 предложений), укажите 3 конкретных шага и 1 ключевой риск. "
            f"Не давайте гарантий доходности. Учитывайте поведенческие ограничения профиля."
        )

# Global instance for import in other modules
llm_coordinator = GigaChatCoordinator()
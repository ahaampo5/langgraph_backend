#!/usr/bin/env python3
"""
백엔드 서버 실행 스크립트
"""

import os
import uvicorn
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

def main():
    """서버 실행"""
    # 환경 변수 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ OPENAI_API_KEY가 설정되지 않았습니다.")
        print("환경 변수를 설정하거나 .env 파일을 생성해주세요.")
        return
    
    # uvicorn으로 FastAPI 서버 실행
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # 개발 모드에서 자동 리로드
        log_level="info"
    )

if __name__ == "__main__":
    main()

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    print("⏳ [1/2] Đang cài đặt các thư viện cần thiết (có thể mất vài phút)...")
    # Danh sách các thư viện cốt lõi của dự án
    packages = [
        "sentence-transformers",
        "faiss-cpu",
        "llama-cpp-python",
        "streamlit",
        "pandas",
        "numpy",
        "huggingface_hub",
        "matplotlib",
        "seaborn",
        "rich",
        "tqdm",
        "openai" # (Nếu bạn có dùng script dùng API)
    ]
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)
        print("✅ Cài đặt thư viện thành công!\n")
    except subprocess.CalledProcessError as e:
        print(f"❌ Lỗi khi cài đặt thư viện: {e}")
        print("Mẹo: Nếu lỗi ở 'llama-cpp-python', bạn hãy thử chạy lệnh:")
        print('pip install llama-cpp-python --prefer-binary --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu')
        sys.exit(1)

def download_embedding_model():
    print("⏳ [2/2] Đang tải model Embedding (BAAI/bge-m3 - khoảng 2.2GB)...")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Lỗi: Chưa cài đặt huggingface_hub.")
        sys.exit(1)

    # Tạo thư mục models nếu chưa có
    base_dir = Path(__file__).parent.resolve()
    model_dir = base_dir / "models" / "bge-m3"
    model_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Tải model về thư mục local
        snapshot_download(
            repo_id="BAAI/bge-m3",
            local_dir=str(model_dir),
            ignore_patterns=["*.jpg", "*.png", "*.webp", "onnx/*", ".gitattributes"] # Bỏ qua ảnh và bản ONNX cho nhẹ
        )
        print(f"✅ Đã tải xong model Embedding vào: {model_dir}\n")
    except Exception as e:
        print(f"❌ Lỗi khi tải model: {e}")
        sys.exit(1)

def print_instructions():
    print("="*60)
    print("🎉 CÀI ĐẶT HOÀN TẤT!")
    print("="*60)
    print("👉 Các bước còn lại bạn cần làm thủ công:")
    print("1. Tải LLM Qwen (nếu chưa có) bằng lệnh:")
    print("   huggingface-cli download Qwen/Qwen2.5-3B-Instruct-GGUF qwen2.5-3b-instruct-q4_k_m.gguf --local-dir qwen_model")
    print("2. Tải Dữ liệu Vector (legal_data.zip) từ link MediaFire trong file README.md.")
    print("   Giải nén và bỏ 2 file (legal_v2.index, legal_meta_v2.json) vào thư mục 'processed/'.")
    print("\n🚀 Sau khi xong 2 bước trên, chạy lệnh sau để mở Chatbot:")
    print("   streamlit run app.py")
    print("="*60)

if __name__ == "__main__":
    print("🚀 BẮT ĐẦU QUÁ TRÌNH CÀI ĐẶT TỰ ĐỘNG...\n")
    install_requirements()
    download_embedding_model()
    print_instructions()
# ask.py — phiên bản đẹp (rich UI)
import json, sys
from pathlib import Path
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt
from rich import box

console = Console()

BASE_DIR   = Path(r"C:\Users\dinht\Downloads\Bo_Cai_Chatbot_Luat")
INDEX_FILE = BASE_DIR / "processed" / "legal.index"
META_FILE  = BASE_DIR / "processed" / "legal_meta.json"
GGUF_FILE  = BASE_DIR / "qwen_model" / "qwen2.5-3b-instruct-q4_k_m.gguf"


def banner():
    console.print(Panel.fit(
        "[bold cyan]⚖️   CHATBOT PHÁP LUẬT VIỆT NAM[/bold cyan]\n"
        "[dim]RAG · Qwen2.5-3B-Instruct · Q4_K_M · CPU[/dim]",
        border_style="cyan", padding=(0, 2),
    ))


def load():
    if not INDEX_FILE.exists():
        console.print(f"[red]✗ Chưa có index:[/red] {INDEX_FILE}")
        console.print("[yellow]→ Chạy 'python build_index.py' xong trước.[/yellow]")
        sys.exit(1)

    with console.status("[cyan]Loading embedder & index...[/cyan]", spinner="dots"):
        embedder = SentenceTransformer("keepitreal/vietnamese-sbert")
        index    = faiss.read_index(str(INDEX_FILE))
        meta     = json.load(open(META_FILE, encoding="utf-8"))
    console.print(f"  [green]✓[/green] {index.ntotal:,} vectors · embedder ready")

    with console.status("[cyan]Loading Qwen GGUF (CPU, 30–60s)...[/cyan]", spinner="dots"):
        llm = Llama(
            model_path=str(GGUF_FILE),
            n_ctx=8192,
            n_threads=8,
            n_gpu_layers=0,
            verbose=False,
        )
    console.print(f"  [green]✓[/green] Qwen ready\n")
    return embedder, index, meta, llm


SYSTEM = (
    "Bạn là trợ lý pháp luật Việt Nam. "
    "Chỉ trả lời dựa trên NGỮ CẢNH được cung cấp. "
    "Nếu ngữ cảnh không có thông tin, trả lời: "
    "'Tôi không tìm thấy thông tin trong cơ sở dữ liệu.' "
    "Khi trả lời, trích dẫn số hiệu văn bản và điều/khoản nếu có."
)


def score_bar(score, width=10):
    n = max(0, min(width, int(round(score * width))))
    return "[green]" + "█" * n + "[/green][dim]" + "░" * (width - n) + "[/dim]"


def ask(embedder, index, meta, llm, question, k=4):
    with console.status("[cyan]Đang tìm ngữ cảnh liên quan...[/cyan]", spinner="dots"):
        q_emb = embedder.encode([question], normalize_embeddings=True).astype("float32")
        scores, ids = index.search(q_emb, k)

    contexts, sources = [], []
    for rank, i in enumerate(ids[0]):
        m = meta[i]
        snippet = m["content"][:1800]
        contexts.append(
            f"[{rank+1}] {m['so_ky_hieu']} — {m['title']}\n"
            f"(Lĩnh vực: {m['linh_vuc']} | Score: {scores[0][rank]:.3f})\n"
            f"{snippet}"
        )
        sources.append({
            "rank": rank+1, "score": float(scores[0][rank]),
            "so_ky_hieu": m["so_ky_hieu"], "title": m["title"],
            "linh_vuc": m["linh_vuc"],
            "doc_id": m["doc_id"], "sample_id": m["sample_id"],
        })

    user_msg = "NGỮ CẢNH:\n" + "\n\n---\n\n".join(contexts) + f"\n\nCÂU HỎI: {question}"
    prompt = (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    console.rule("[bold yellow]🤖  Trả lời[/bold yellow]", style="yellow")
    console.print()
    buf = []
    for chunk in llm(prompt, max_tokens=600, temperature=0.2,
                     stop=["<|im_end|>", "<|im_start|>"], stream=True):
        piece = chunk["choices"][0]["text"]
        if piece:
            console.print(piece, end="", markup=False, highlight=False, soft_wrap=True)
            buf.append(piece)
    console.print("\n")
    console.rule(style="yellow")
    console.print()

    return "".join(buf).strip(), sources


def show_sources(sources):
    t = Table(
        title="[bold cyan]📚  Nguồn tham khảo[/bold cyan]",
        box=box.ROUNDED, border_style="cyan",
        show_lines=False, pad_edge=False, title_justify="left",
    )
    t.add_column("#", justify="right", style="dim", width=2)
    t.add_column("Điểm", justify="center", width=16)
    t.add_column("Số hiệu", style="bold green", no_wrap=True)
    t.add_column("Tiêu đề", overflow="ellipsis", max_width=55)
    t.add_column("Lĩnh vực", style="dim", overflow="ellipsis", max_width=18)
    for s in sources:
        t.add_row(
            str(s["rank"]),
            f"{s['score']:.3f} {score_bar(s['score'])}",
            s["so_ky_hieu"] or "—",
            s["title"],
            s["linh_vuc"] or "—",
        )
    console.print(t)


def main():
    banner()
    embedder, index, meta, llm = load()

    # Chế độ 1 lần: python ask.py "câu hỏi"
    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
        console.print(Panel(q, title="[bold magenta]❓  Câu hỏi[/bold magenta]",
                            border_style="magenta", box=box.ROUNDED, padding=(0, 1)))
        console.print()
        ans, srcs = ask(embedder, index, meta, llm, q)
        show_sources(srcs)
        return

    # Chat loop
    console.print("[dim]Nhập câu hỏi — gõ [bold]exit[/bold] để thoát[/dim]\n")
    while True:
        try:
            q = Prompt.ask("[bold magenta]Bạn[/bold magenta]")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Tạm biệt 👋[/dim]")
            break
        q = q.strip()
        if not q or q.lower() in ("exit", "quit", "q"):
            console.print("[dim]Tạm biệt 👋[/dim]")
            break
        ans, srcs = ask(embedder, index, meta, llm, q)
        show_sources(srcs)
        console.print()


if __name__ == "__main__":
    main()
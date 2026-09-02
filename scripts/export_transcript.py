#!/usr/bin/env python3
"""
OmniQuery-AI: Antigravity Session Transcript Exporter
Utility to parse and export raw Antigravity CLI (.gemini/antigravity-cli) conversation transcripts
into human-readable plain text (.txt) files.
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime


def get_brain_dir() -> Path:
    home = Path.home()
    return home / ".gemini" / "antigravity-cli" / "brain"


def find_latest_conversation(brain_dir: Path) -> Path:
    if not brain_dir.exists():
        raise FileNotFoundError(f"Brain directory not found at {brain_dir}")

    conv_dirs = [d for d in brain_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    if not conv_dirs:
        raise FileNotFoundError(f"No conversation sessions found in {brain_dir}")

    # Sort by modification time (most recent first)
    conv_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return conv_dirs[0]


def export_transcript(conv_id: str = None, output_filename: str = None):
    brain_dir = get_brain_dir()

    if conv_id:
        target_dir = brain_dir / conv_id
    else:
        target_dir = find_latest_conversation(brain_dir)
        conv_id = target_dir.name

    transcript_path = target_dir / ".system_generated" / "logs" / "transcript_full.jsonl"
    if not transcript_path.exists():
        # Fallback to standard transcript.jsonl
        transcript_path = target_dir / ".system_generated" / "logs" / "transcript.jsonl"

    if not transcript_path.exists():
        print(f"❌ Error: No transcript log found for conversation {conv_id}")
        sys.exit(1)

    if not output_filename:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"OmniQuery_AI_Transcript_{conv_id[:8]}_{timestamp_str}.txt"

    output_path = Path.cwd() / output_filename

    print(f"📖 Reading transcript from: {transcript_path}")
    print(f"✍️  Exporting to: {output_path}")

    with open(transcript_path, "r", encoding="utf-8") as f_in, open(output_path, "w", encoding="utf-8") as f_out:
        f_out.write("=" * 80 + "\n")
        f_out.write("OMNIQUERY-AI: CONVERSATION TRANSCRIPT EXPORT\n")
        f_out.write(f"Conversation ID: {conv_id}\n")
        f_out.write(f"Exported At: {datetime.now().isoformat()}\n")
        f_out.write("=" * 80 + "\n\n")

        line_count = 0
        for line in f_in:
            line_count += 1
            try:
                step = json.loads(line)
            except Exception:
                continue

            step_idx = step.get("step_index", "")
            step_type = step.get("type", "")
            created_at = step.get("created_at", "")
            content = step.get("content", "")
            tool_calls = step.get("tool_calls", [])

            if step_type == "USER_INPUT":
                f_out.write("\n" + "=" * 80 + "\n")
                f_out.write(f">>> USER PROMPT | Step #{step_idx} | {created_at}\n")
                f_out.write("=" * 80 + "\n")
                f_out.write(f"{content}\n\n")

            elif step_type == "PLANNER_RESPONSE":
                if content:
                    f_out.write("\n" + "-" * 60 + "\n")
                    f_out.write(f"<<< MODEL RESPONSE | Step #{step_idx} | {created_at}\n")
                    f_out.write("-" * 60 + "\n")
                    f_out.write(f"{content}\n\n")

                if tool_calls:
                    for tc in tool_calls:
                        fn_name = tc.get("name", "tool")
                        fn_args = tc.get("args", {})
                        f_out.write(f"[TOOL INVOCATION] {fn_name}:\n")
                        f_out.write(f"{json.dumps(fn_args, indent=2, ensure_ascii=False)}\n\n")

            elif step_type == "GENERIC":
                if content:
                    f_out.write(f"[TOOL EXECUTION OUTPUT] | Step #{step_idx}\n")
                    f_out.write(f"{content}\n\n")

            elif step_type == "SYSTEM_MESSAGE":
                if content:
                    f_out.write(f"[SYSTEM NOTIFICATION] | Step #{step_idx}\n")
                    f_out.write(f"{content}\n\n")

    file_size_kb = output_path.stat().st_size / 1024
    print(f"✅ Successfully exported {line_count} log steps ({file_size_kb:.1f} KB) to:")
    print(f"   {output_path.resolve()}")


if __name__ == "__main__":
    cid = sys.argv[1] if len(sys.argv) > 1 else None
    out = sys.argv[2] if len(sys.argv) > 2 else None
    export_transcript(cid, out)

from worldclaw_oss.cli import DEFAULT_LIVE_PROMPT_FILE, parser


def test_live_generate_prompt_file_is_optional_and_canonical():
    args = parser().parse_args(["generate", "--mode", "live"])
    assert args.prompt_file is None
    assert DEFAULT_LIVE_PROMPT_FILE.is_file()
    assert DEFAULT_LIVE_PROMPT_FILE.read_text(encoding="utf-8").strip() == (
        "A forest region with lakes, rivers, cabins and trails"
    )


def test_synthetic_generate_prompt_file_remains_explicit():
    args = parser().parse_args(["generate", "--mode", "synthetic"])
    assert args.prompt_file is None

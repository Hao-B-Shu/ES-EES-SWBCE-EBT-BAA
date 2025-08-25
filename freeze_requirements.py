from pipreqs.pipreqs import init

def freeze_dependencies():#存储依赖包版本
    args = {
        "<path>": ".",
        "--encoding": "utf-8",
        "--use-local": False,
        "--ignore": [],
        "--ignore-dir": [],
        "--skip": [],
        "--force": True,
        "--proxy": None,
        "--savepath": "requirements.freeze.txt",
        "--print": False,
        "--diff": False,
        "--debug": False,
        "--pypi-server": "https://pypi.org/pypi",
        "--clean": False,
        "--no-pin": False,
        "--compile": False,
        "--extra": None,
        "--mode": "compat",  # ✅ 合法模式值
        "--success-only": False,
        "--follow-links": False,
        "--encoding-errors": "strict"
    }
    init(args)
    print("✅ 依赖已记录到: requirements.freeze.txt")

if __name__ == "__main__":
    freeze_dependencies()

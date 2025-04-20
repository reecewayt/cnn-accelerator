# cnn-accelerator
A CNN accelerator based in MyHDL (Python)

### Install required packages

```bash
python -m venv venv

source venv/bin/activate

# Install packages from requirements.txt
pip install -r requirements.txt
```


### Project Structure

```bash
cnn-accelerator/
├── src/
│   ├── __init__.py
│   ├── hdl/                  # MyHDL source code
│   │   ├── __init__.py
│   │   ├── layers/           # CNN layer implementations
│   │   ├── memory/           # Memory interfaces
│   │   └── top.py            # Top-level design
│   └── utils/                # Helper functions
│       └── __init__.py
├── gen/                      # Generated Verilog/VHDL files
│   └── verilog/              # Generated Verilog output
├── tests/                    # Test files
│   ├── unit/                 # Unit tests for individual components
│   └── integration/          # Full system tests
├── docs/                     # Documentation
├── scripts/                  # Build scripts, automation
│   └── generate_hdl.py       # Script to run code generation
├── venv/                     # Virtual environment (gitignored)
├── requirements.txt          # Dependencies
└── README.md                 # Project documentation
```

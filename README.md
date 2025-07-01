# MCP Server for Financial Analysis

## Introduction
This project is a **Model Context Protocol (MCP) server** designed to enable LLMs (such as Claude, Cursor, or GPT-based agents) to retrieve, analyze, and visualize stock prices and financial report data. It provides a set of robust tools for quantitative trading research, investment analysis, and financial education.

## Usage Scenarios
- **LLM-driven trading analysis**: Let LLMs fetch and analyze stock data, financial statements, and technical indicators to generate trading insights.
- **Financial metric calculation**: Compute key ratios and metrics from income statements, balance sheets, and cash flow tables.
- **Visualization**: Generate price charts, financial metric trends, and highlight trading opportunities.
- **Automated research**: Integrate with LLMs to answer questions like "What is the ROE of AAPL?", "Show me the last 6 months of MSFT price data.", or "Plot buy/sell signals for TSLA."

## Features
- Retrieve real-time and historical stock prices (single or multiple tickers)
- Extract and analyze financial statements (income, balance sheet, cash flow)
- Calculate key financial and technical metrics (PE, ROE, moving averages, etc.)
- Visualize price data, metrics, and trading signals
- LLM-friendly, JSON-serializable outputs

## Installation
1. **Clone the repository**
   ```bash
   git clone https://github.com/Nishaant-Soni/MCP_Server_for_Financial_Analysis.git
   cd <your-repo-folder>
   ```
2. **Set up a Python virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. **Install dependencies**
   ```bash
   uv sync
   ```

## How to Run
1. **Configure your LLM client (Claude, Cursor, etc.)** to connect to the MCP server and call the available tools.

    Go to Claude settings
    Select Developer --> Edit Config
    Add new MCP server.
    In the JSON File, add
    ```
    {"mcpServers": 
         {"trader-mcp": {
                "command": "uv",
                "args": [
                "--directory",
                "/directory/to/your/mcp",
                "run",
                "main.py"
            ]
            }
         }
    }  
    ```   
'trader-mcp' should now be listed in your Claude tools.

2. **Start talking with your trading assistant!** 

Quries like:

Compare the stock prices of Nvidia and AMD in the past month.

How's Tesla stock like in the past 3 months?

Plot the trading opportunities of Microsoft in the past 3 months.

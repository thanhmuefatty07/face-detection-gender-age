#!/usr/bin/env python3
"""Convert Markdown to styled HTML"""

import markdown
from pathlib import Path

# Read markdown
md_path = Path("TECHNICAL_REPORT.md")
md_content = md_path.read_text(encoding='utf-8')

# Convert to HTML
html_content = markdown.markdown(
    md_content,
    extensions=['tables', 'fenced_code', 'toc']
)

# Add styling
styled_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Báo Cáo Kỹ Thuật - Face Detection & Gender/Age</title>
    <style>
        @media print {{
            @page {{
                size: A4;
                margin: 1.5cm;
            }}
            body {{
                font-size: 10pt;
            }}
            pre {{
                page-break-inside: avoid;
            }}
            h2, h3 {{
                page-break-after: avoid;
            }}
        }}
        
        * {{
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', 'Roboto', Arial, sans-serif;
            font-size: 11pt;
            line-height: 1.7;
            color: #2c3e50;
            max-width: 900px;
            margin: 0 auto;
            padding: 40px 30px;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
        }}
        
        .container {{
            background: white;
            padding: 50px;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        }}
        
        h1 {{
            color: #1a5276;
            border-bottom: 4px solid #3498db;
            padding-bottom: 15px;
            font-size: 28pt;
            text-align: center;
            margin-bottom: 10px;
        }}
        
        h1 + p {{
            text-align: center;
            color: #7f8c8d;
            font-size: 14pt;
            margin-bottom: 40px;
        }}
        
        h2 {{
            color: #2874a6;
            border-left: 5px solid #3498db;
            padding-left: 15px;
            margin-top: 40px;
            font-size: 18pt;
            background: linear-gradient(90deg, #ebf5fb 0%, transparent 100%);
            padding: 10px 15px;
            border-radius: 0 8px 8px 0;
        }}
        
        h3 {{
            color: #2e86c1;
            font-size: 14pt;
            margin-top: 25px;
            padding-bottom: 5px;
            border-bottom: 2px dashed #aed6f1;
        }}
        
        h4 {{
            color: #5dade2;
            font-size: 12pt;
            margin-top: 20px;
        }}
        
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            font-size: 10pt;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-radius: 8px;
            overflow: hidden;
        }}
        
        th, td {{
            border: 1px solid #e8e8e8;
            padding: 12px 15px;
            text-align: left;
        }}
        
        th {{
            background: linear-gradient(135deg, #2874a6 0%, #1a5276 100%);
            color: white;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 9pt;
            letter-spacing: 0.5px;
        }}
        
        tr:nth-child(even) {{
            background-color: #f8f9fa;
        }}
        
        tr:hover {{
            background-color: #e8f6fc;
            transition: background-color 0.3s;
        }}
        
        code {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 10pt;
            font-weight: 600;
        }}
        
        pre {{
            background: linear-gradient(135deg, #2c3e50 0%, #1a252f 100%);
            color: #ecf0f1;
            padding: 20px;
            border-radius: 10px;
            overflow-x: auto;
            font-size: 9pt;
            line-height: 1.5;
            box-shadow: 0 5px 20px rgba(0,0,0,0.2);
        }}
        
        pre code {{
            background: none;
            -webkit-text-fill-color: #ecf0f1;
            color: #ecf0f1;
            font-weight: normal;
        }}
        
        blockquote {{
            border-left: 5px solid #3498db;
            margin: 20px 0;
            padding: 15px 25px;
            background: linear-gradient(90deg, #ebf5fb 0%, #fff 100%);
            border-radius: 0 10px 10px 0;
            font-style: italic;
            color: #34495e;
        }}
        
        blockquote strong {{
            color: #2874a6;
            font-style: normal;
        }}
        
        strong {{
            color: #1a5276;
        }}
        
        hr {{
            border: none;
            height: 3px;
            background: linear-gradient(90deg, #3498db, #9b59b6, #e74c3c);
            margin: 40px 0;
            border-radius: 3px;
        }}
        
        ul, ol {{
            padding-left: 25px;
        }}
        
        li {{
            margin: 8px 0;
        }}
        
        /* Emoji styling */
        .emoji {{
            font-size: 1.2em;
        }}
        
        /* Print button */
        .print-btn {{
            position: fixed;
            top: 20px;
            right: 20px;
            background: linear-gradient(135deg, #3498db 0%, #2874a6 100%);
            color: white;
            border: none;
            padding: 15px 25px;
            border-radius: 30px;
            cursor: pointer;
            font-size: 12pt;
            font-weight: 600;
            box-shadow: 0 5px 20px rgba(52, 152, 219, 0.4);
            transition: transform 0.3s, box-shadow 0.3s;
        }}
        
        .print-btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(52, 152, 219, 0.5);
        }}
        
        @media print {{
            .print-btn {{
                display: none;
            }}
            body {{
                background: white;
                padding: 0;
            }}
            .container {{
                box-shadow: none;
                padding: 0;
            }}
        }}
    </style>
</head>
<body>
    <button class="print-btn" onclick="window.print()">🖨️ In PDF</button>
    <div class="container">
        {html_content}
    </div>
</body>
</html>
"""

# Save HTML
output_path = Path("TECHNICAL_REPORT.html")
output_path.write_text(styled_html, encoding='utf-8')

print(f"✅ HTML created: {{output_path.absolute()}}")
print("📌 Mở file trong trình duyệt → Ctrl+P → Save as PDF")

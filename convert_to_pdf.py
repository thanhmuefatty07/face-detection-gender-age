#!/usr/bin/env python3
"""Convert Markdown to styled HTML for PDF printing"""

import markdown
from pathlib import Path

# Read markdown
md_path = Path("TECHNICAL_REPORT.md")
md_content = md_path.read_text(encoding='utf-8')

# Convert to HTML
html_content = markdown.markdown(
    md_content,
    extensions=['tables', 'fenced_code', 'toc', 'codehilite']
)

# Add professional styling
styled_html = f"""
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="utf-8">
    <title>Báo Cáo Kỹ Thuật - Face Detection & Gender/Age</title>
    <style>
        @media print {{
            @page {{
                size: A4;
                margin: 1.5cm;
            }}
            body {{ font-size: 9pt; }}
            pre {{ page-break-inside: avoid; }}
            h2, h3 {{ page-break-after: avoid; }}
            .print-btn {{ display: none; }}
        }}
        
        * {{ box-sizing: border-box; }}
        
        body {{
            font-family: 'Segoe UI', 'Roboto', Arial, sans-serif;
            font-size: 11pt;
            line-height: 1.7;
            color: #2c3e50;
            max-width: 1000px;
            margin: 0 auto;
            padding: 40px 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        
        .container {{
            background: white;
            padding: 50px;
            border-radius: 15px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }}
        
        h1 {{
            color: #1a1a2e;
            border-bottom: 4px solid #667eea;
            padding-bottom: 15px;
            font-size: 26pt;
            text-align: center;
            margin-bottom: 10px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        
        h2 {{
            color: #16213e;
            border-left: 5px solid #667eea;
            padding-left: 15px;
            margin-top: 40px;
            font-size: 16pt;
            background: linear-gradient(90deg, #f0f4ff 0%, transparent 100%);
            padding: 12px 15px;
            border-radius: 0 8px 8px 0;
        }}
        
        h3 {{
            color: #1a1a2e;
            font-size: 13pt;
            margin-top: 25px;
            padding-bottom: 5px;
            border-bottom: 2px dashed #e0e0e0;
        }}
        
        h4 {{
            color: #4a4a4a;
            font-size: 11pt;
            margin-top: 20px;
        }}
        
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            font-size: 9pt;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            border-radius: 10px;
            overflow: hidden;
        }}
        
        th, td {{
            border: 1px solid #e8e8e8;
            padding: 10px 12px;
            text-align: left;
        }}
        
        th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 8pt;
            letter-spacing: 0.5px;
        }}
        
        tr:nth-child(even) {{
            background-color: #f8f9ff;
        }}
        
        tr:hover {{
            background-color: #e8ecff;
            transition: background-color 0.3s;
        }}
        
        code {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 9pt;
            font-weight: 600;
        }}
        
        pre {{
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #e0e0e0;
            padding: 20px;
            border-radius: 10px;
            overflow-x: auto;
            font-size: 8pt;
            line-height: 1.5;
            box-shadow: 0 8px 25px rgba(0,0,0,0.2);
        }}
        
        pre code {{
            background: none;
            -webkit-text-fill-color: #e0e0e0;
            color: #e0e0e0;
            font-weight: normal;
        }}
        
        blockquote {{
            border-left: 5px solid #667eea;
            margin: 20px 0;
            padding: 15px 25px;
            background: linear-gradient(90deg, #f0f4ff 0%, white 100%);
            border-radius: 0 10px 10px 0;
            font-style: italic;
            color: #34495e;
        }}
        
        strong {{
            color: #16213e;
        }}
        
        hr {{
            border: none;
            height: 3px;
            background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
            margin: 40px 0;
            border-radius: 3px;
        }}
        
        ul, ol {{
            padding-left: 25px;
        }}
        
        li {{
            margin: 6px 0;
        }}
        
        .print-btn {{
            position: fixed;
            top: 20px;
            right: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 30px;
            cursor: pointer;
            font-size: 12pt;
            font-weight: 600;
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
            transition: all 0.3s;
        }}
        
        .print-btn:hover {{
            transform: translateY(-3px);
            box-shadow: 0 12px 35px rgba(102, 126, 234, 0.5);
        }}
        
        /* Checkmarks and crosses */
        td:contains("✅"), td:contains("❌"), td:contains("⚠️") {{
            font-size: 12pt;
        }}
    </style>
</head>
<body>
    <button class="print-btn" onclick="window.print()">🖨️ Xuất PDF</button>
    <div class="container">
        {html_content}
    </div>
</body>
</html>
"""

# Save HTML
output_path = Path("TECHNICAL_REPORT.html")
output_path.write_text(styled_html, encoding='utf-8')

print(f"✅ HTML created: {output_path.absolute()}")
print("📌 Mở file trong trình duyệt → Ctrl+P → Save as PDF")

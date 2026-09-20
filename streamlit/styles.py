"""
styles.py — Modern UI/UX Design System for the Collision Severity Dashboard
Provides custom CSS tokens, glassmorphism cards, metric badges, and color palettes.
"""

def get_custom_css():
    return """
    <style>
        /* Import Google Font Inter */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* Gradient header card */
        .hero-banner {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f766e 100%);
            color: #ffffff;
            padding: 2.2rem 2rem;
            border-radius: 16px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3), 0 8px 10px -6px rgba(0, 0, 0, 0.3);
            margin-bottom: 2rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .hero-title {
            font-size: 2.2rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            margin-bottom: 0.5rem;
            background: linear-gradient(to right, #ffffff, #67e8f9);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .hero-subtitle {
            font-size: 1.05rem;
            color: #cbd5e1;
            max-width: 850px;
            line-height: 1.6;
        }

        /* Glassmorphism Info Cards */
        .glass-card {
            background: rgba(30, 41, 59, 0.7);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 14px;
            padding: 1.5rem;
            margin-bottom: 1.2rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        }

        /* Step flow pills */
        .flow-pill {
            display: inline-block;
            padding: 0.35rem 0.85rem;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
        }

        .flow-pill-blue { background: rgba(59, 130, 246, 0.2); color: #93c5fd; border: 1px solid rgba(59, 130, 246, 0.4); }
        .flow-pill-purple { background: rgba(168, 85, 247, 0.2); color: #d8b4fe; border: 1px solid rgba(168, 85, 247, 0.4); }
        .flow-pill-emerald { background: rgba(16, 185, 129, 0.2); color: #6ee7b7; border: 1px solid rgba(16, 185, 129, 0.4); }
        .flow-pill-amber { background: rgba(245, 158, 11, 0.2); color: #fde68a; border: 1px solid rgba(245, 158, 11, 0.4); }

        /* Severity Badges */
        .severity-badge-fatal {
            background: linear-gradient(135deg, #7f1d1d 0%, #dc2626 100%);
            color: #ffffff;
            font-size: 1.6rem;
            font-weight: 800;
            padding: 1.2rem 2rem;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 0 20px rgba(220, 38, 38, 0.4);
            border: 1px solid #ef4444;
            animation: pulse-red 2s infinite;
        }

        .severity-badge-serious {
            background: linear-gradient(135deg, #7c2d12 0%, #ea580c 100%);
            color: #ffffff;
            font-size: 1.6rem;
            font-weight: 800;
            padding: 1.2rem 2rem;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 0 20px rgba(234, 88, 12, 0.35);
            border: 1px solid #f97316;
        }

        .severity-badge-slight {
            background: linear-gradient(135deg, #14532d 0%, #16a34a 100%);
            color: #ffffff;
            font-size: 1.6rem;
            font-weight: 800;
            padding: 1.2rem 2rem;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 0 20px rgba(22, 163, 74, 0.35);
            border: 1px solid #22c55e;
        }

        /* Metric Highlights */
        .metric-card {
            background: #1e293b;
            border-radius: 12px;
            padding: 1.2rem;
            border: 1px solid #334155;
            text-align: center;
        }

        .metric-val {
            font-size: 1.8rem;
            font-weight: 700;
            color: #38bdf8;
            margin-top: 0.2rem;
        }

        .metric-lbl {
            font-size: 0.85rem;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        /* Formula box */
        .math-box {
            background: #0f172a;
            border-left: 4px solid #06b6d4;
            padding: 1rem 1.4rem;
            border-radius: 0 8px 8px 0;
            margin: 1rem 0;
            font-family: 'Courier New', Courier, monospace;
        }

        /* Highlight table */
        table {
            border-collapse: separate;
            border-spacing: 0;
            width: 100%;
            border-radius: 8px;
            overflow: hidden;
        }

        th {
            background-color: #1e293b !important;
            color: #94a3b8 !important;
            font-weight: 600;
            padding: 10px 14px !important;
            text-align: left;
        }

        td {
            padding: 10px 14px !important;
            border-bottom: 1px solid #334155 !important;
        }
    </style>
    """

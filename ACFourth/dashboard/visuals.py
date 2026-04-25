import plotly.graph_objs as go
from dash import html
from config import BUS_COORDS, LINE_PARAMS

def create_sld_figure(bus_data, branch_data, meas_data=None):
    
    fig = go.Figure()
    annotations = []

    # ==========================================
    # วาดสายส่ง (Lines) และใส่ค่า Line Flows
    # ==========================================
    for (bus_a, bus_b), params in LINE_PARAMS.items():
        x0, y0 = BUS_COORDS[bus_a]
        x1, y1 = BUS_COORDS[bus_b]
        
        # วาดเส้น
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1], mode='lines',
            line=dict(color='#888888', width=4), hoverinfo='none', showlegend=False
        ))
        
        # ดึงค่า Flow อย่างปลอดภัย (ป้องกัน KeyError)
        flow_p, flow_q = 0.0, 0.0
        branch_id = None
        
        if (bus_a, bus_b) == ('Bus 1', 'Bus 2'): branch_id = 1
        elif (bus_a, bus_b) == ('Bus 1', 'Bus 3'): branch_id = 2
        elif (bus_a, bus_b) in [('Bus 3', 'Bus 2'), ('Bus 2', 'Bus 3')]: branch_id = 3

        if branch_id and branch_id in branch_data:
            b_info = branch_data[branch_id]
            fbus = int(bus_a[-1:])
            tbus = int(bus_b[-1:])
            key_p = f"p{fbus}{tbus}_from"
            key_q = f"q{fbus}{tbus}_from"
            # ใช้ .get() ต่อกัน ถ้าไม่มี key ให้คืนค่า 0.0 แทนการล่ม
            flow_p = b_info[key_p]["value"] if key_p in b_info else 0.0
            flow_q = b_info[key_q]["value"] if key_q in b_info else 0.0

        mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2

        # ขยับกล่องข้อความไม่ให้ทับเส้น
        if (bus_a, bus_b) == ('Bus 1', 'Bus 2'): mid_y += 0.6 
        elif (bus_a, bus_b) == ('Bus 1', 'Bus 3'): mid_x -= 1.0; mid_y -= 0.05 
        elif (bus_a, bus_b) in [('Bus 2', 'Bus 3'), ('Bus 3', 'Bus 2')]: mid_x += 1.0; mid_y -= 0.05 
        
        annotation_text = (
            f"X = {params.get('X', 'N/A')} p.u.<br>"
            f"<span style='color:#2ca02c; font-weight:bold'>P = {flow_p:.2f} W</span><br>"
            f"<span style='color:#ff7f0e; font-weight:bold'>Q = {flow_q:.2f} VAR</span>"
        )
        annotations.append(dict(
            x=mid_x, y=mid_y, text=annotation_text, showarrow=False, 
            font=dict(size=12, family="Arial"), bgcolor="white", bordercolor="#d9534f", borderwidth=1, borderpad=3
        ))

    # ==========================================
    # สัญลักษณ์ Generator & Load
    # ==========================================
    fig.add_shape(type="circle", x0=0.75, y0=9.7, x1=1.35, y1=10.3, line_color="black", line_width=2, fillcolor="white")
    annotations.append(dict(x=1.05, y=10, text="~", showarrow=False, font=dict(size=22, color="black")))
    fig.add_trace(go.Scatter(x=[1.3, 2], y=[10, 10], mode='lines', line=dict(color='black', width=2), showlegend=False))

    annotations.append(dict(x=11.5, y=10, ax=10.4, ay=10, xref="x", yref="y", axref="x", ayref="y", text="Load (P2)", showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=2, font=dict(size=14, color="blue")))
    annotations.append(dict(x=6, y=4.5, ax=6, ay=5.6, xref="x", yref="y", axref="x", ayref="y", text="Load (P3)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;", showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=2, font=dict(size=14, color="blue")))

    # ==========================================
    # วาด Bus (Nodes) และค่า V, P, Q
    # ==========================================
    x_nodes, y_nodes, text_nodes = [], [], []
    text_positions = ["top center", "top center", "bottom right"]

    for bus_str, coords in BUS_COORDS.items():
        x_nodes.append(coords[0])
        y_nodes.append(coords[1])
        
        # แปลงชื่อ 'Bus 1' เป็นเลข 1
        bus_id = int(bus_str.replace('Bus ', ''))
        
        # ดึงค่าอย่างปลอดภัย
        b_data = bus_data.get(bus_id, {})
        meas_data_bus = meas_data.get(bus_id, {}) if meas_data else {}
        v_pu = b_data.get(f'v{bus_id}', {}).get('value_pu', 0.0)
        ang_deg = b_data.get('vang_deg', {}).get('value', 0.0)
        p_meas = meas_data_bus.get(f'p{bus_id}', {}).get('value', 0.0)
        q_meas = meas_data_bus.get(f'q{bus_id}', {}).get('value', 0.0)
        p_est = b_data.get(f'p{bus_id}', {}).get('value', 0.0)
        q_est = b_data.get(f'q{bus_id}', {}).get('value', 0.0)
        
        text_node = (f"<b>{bus_str}</b><br>"
                     f"V: {v_pu:.4f} p.u. ∠ {ang_deg:.2f}°<br>"
                     f"P: {p_meas:.2f} W | <span style='color:green'>Est: {p_est:.2f}</span> W<br>"
                     f"Q: {q_meas:.2f} VAR | <span style='color:#ff7f0e'>Est: {q_est:.2f}</span> VAR")
        text_nodes.append(text_node)

    fig.add_trace(go.Scatter(
        x=x_nodes, y=y_nodes, mode='markers+text',
        marker=dict(size=30, color='rgba(0,0,0,0)', symbol='line-ns', line=dict(width=5, color='black')), 
        text=text_nodes, textposition=text_positions, textfont=dict(size=13, color='black'), showlegend=False
    ))

    # ==========================================
    # Layout
    # ==========================================
    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1, 13]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[3, 12]),
        plot_bgcolor='white', paper_bgcolor='white', margin=dict(l=20, r=20, t=20, b=20),
        annotations=annotations 
    )
    return fig


def create_detail_cards(row_data):
    if not row_data:
        return html.Div("Click on a data point in the table to view detailed information", 
                        style={'color': '#888', 'textAlign': 'center', 'padding': '30px', 'fontSize': '18px'})

    time_str = row_data.get('time', '-')
    
    card_style = {'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '10px', 
                  'boxShadow': '0 4px 8px rgba(0,0,0,0.1)', 'flex': '1', 'margin': '0 10px'}

    return html.Div([
        html.H4(f"In-depth analysis of the system status at the time {time_str}", style={'color': '#2c3e50', 'marginLeft': '10px'}),
        
        html.Div([
            # ==============================
            # CARD 1: Bus Parameters 
            # ==============================
            html.Div([
                html.H5("Bus Power (Estimated)", style={'color': '#17a2b8', 'borderBottom': '2px solid #17a2b8', 'paddingBottom': '10px'}),
                
                html.Div([
                    html.B("Bus 1 (Slack): "), html.Br(),
                    html.Span(f"P = {row_data.get('p1_est', 0):.2f} W | ", style={'color': 'green'}),
                    html.Span(f"Q = {row_data.get('q1_est', 0):.2f} VAR", style={'color': '#ff7f0e'})
                ], style={'marginBottom': '10px', 'borderBottom': '1px dashed #eee', 'paddingBottom': '5px'}),
                
                html.Div([
                    html.B("Bus 2 (Load): "), html.Br(),
                    html.Span(f"P = {row_data.get('p2_est', 0):.2f} W | ", style={'color': 'green'}),
                    html.Span(f"Q = {row_data.get('q2_est', 0):.2f} VAR", style={'color': '#ff7f0e'})
                ], style={'marginBottom': '10px', 'borderBottom': '1px dashed #eee', 'paddingBottom': '5px'}),
                
                html.Div([
                    html.B("Bus 3 (Load): "), html.Br(),
                    html.Span(f"P = {row_data.get('p3_est', 0):.2f} W | ", style={'color': 'green'}),
                    html.Span(f"Q = {row_data.get('q3_est', 0):.2f} VAR", style={'color': '#ff7f0e'})
                ])
            ], style=card_style),

            # ==============================
            # CARD 2: Measurement Error
            # ==============================
            html.Div([
                html.H5("Measurement Error (%)", style={'color': '#ffc107', 'borderBottom': '2px solid #ffc107', 'paddingBottom': '10px'}),
                
                html.Div([
                    html.B("Bus 1: "), html.Br(),
                    html.Span(f"P Error = {row_data.get('p1_err', 0):.2f}% | Q Error = {row_data.get('q1_err', 0):.2f}%")
                ], style={'marginBottom': '15px'}),
                
                html.Div([
                    html.B("Bus 2: "), html.Br(),
                    html.Span(f"P Error = {row_data.get('p2_err', 0):.2f}% | Q Error = {row_data.get('q2_err', 0):.2f}%")
                ], style={'marginBottom': '15px'}),
                
                html.Div([
                    html.B("Bus 3: "), html.Br(),
                    html.Span(f"P Error = {row_data.get('p3_err', 0):.2f}% | Q Error = {row_data.get('q3_err', 0):.2f}%")
                ])
            ], style=card_style),

            # ==============================
            # CARD 3: System Health
            # ==============================
            html.Div([
                html.H5("System Health", style={'color': '#28a745', 'borderBottom': '2px solid #28a745', 'paddingBottom': '10px'}),
                html.P([html.B("Algorithm: "), "WLS (AC State Estimation)"]),
                html.P([
                    html.B("Status: "), 
                    html.Span("Real-Time Tracking", style={'color': 'green', 'fontWeight': 'bold'})
                ]),
                html.P([html.B("Timestamp: "), time_str]),
            ], style=card_style),
            
        ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginTop': '15px'})
    ])
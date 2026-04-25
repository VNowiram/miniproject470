import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import random
import math
from dash import dash_table 
from datetime import datetime 
from dash.dependencies import Input, Output, State
from dash import ctx
#from calculations import fetch_meter_data, compute_power_flow, update_table_data,set_meter_mode
from visuals import create_sld_figure, create_detail_cards
import webview
from threading import Thread
import multiprocessing 
import tempfile

from runtime import System 
from measurement.modbusconv import ModbusMeter
from measurement.meas_manager import ParallelModbus


app = dash.Dash(__name__)
app.config.suppress_callback_exceptions = True
history_data = []

"""
meter_map = {"bus1" : '192.168.1.20',
             "bus3" : '192.168.1.21',
             "bus2" : '192.168.1.22',
             "branch3" : '192.168.1.201',
             } 

meters = [ModbusMeter(meter_map["bus1"]),
    ModbusMeter(meter_map["bus3"]),
    ModbusMeter(meter_map["bus2"]),
    ModbusMeter(meter_map["branch3"])
    ]

group = ParallelModbus(meters)
group.connect_all()
"""

grid = System()
grid.add_bus(id=1, name="Slack", type="slack", slack=True)
grid.add_bus(id=2, name="Bus2",  type="PQ")
grid.add_bus(id=3, name="Bus3",  type="PQ")
grid.add_branch(id=1, fbus=1, tbus=2, rs=0.035, xs=0.25, xsh=0.00)
grid.add_branch(id=2, fbus=1, tbus=3, rs=0.035, xs=0.25, xsh=0.00)
grid.add_branch(id=3, fbus=2, tbus=3, rs=0.035, xs=0.25, xsh=0.00)
grid.add_generator(id=1, name="Gen1", gbus=1, pg=0.2148315, qg=0.000171)
grid.add_load(id=1, name="Load1", lbus=2)
grid.add_load(id=2, name="Load2", lbus=3)


def create_sld_tab():
    
    return html.Div([
        dcc.Graph(id='sld-graph', style={'height': '600px'}),
        
    ], style={'paddingTop': '20px'})


def create_report_tab():
    return html.Div([
        html.H3("Data comparison table", style={'textAlign': 'center', 'fontFamily': 'Arial'}),
        
        dash_table.DataTable(
            id='data-table', 
            columns=[
                {"name": "เวลา", "id": "time"}, # ย่อชื่อให้สั้นลง
                {"name": "P1 Meas", "id": "p1_meas"},
                {"name": "P1 Est", "id": "p1_est"},
                {"name": "P1 Err(%)", "id": "p1_err"},
                {"name": "Q1 Meas", "id": "q1_meas"},
                {"name": "Q1 Est", "id": "q1_est"},
                {"name": "Q1 Err(%)", "id": "q1_err"},
            
                {"name": "P2 Meas", "id": "p2_meas"},
                {"name": "P2 Est", "id": "p2_est"},
                {"name": "P2 Err(%)", "id": "p2_err"},
                {"name": "Q2 Meas", "id": "q2_meas"},
                {"name": "Q2 Est", "id": "q2_est"},
                {"name": "Q2 Err(%)", "id": "q2_err"},

                {"name": "P3 Meas", "id": "p3_meas"},
                {"name": "P3 Est", "id": "p3_est"},
                {"name": "P3 Err(%)", "id": "p3_err"},
                {"name": "Q3 Meas", "id": "q3_meas"},
                {"name": "Q3 Est", "id": "q3_est"},
                {"name": "Q3 Err(%)", "id": "q3_err"}
            ],
            data=[],
            
            # 1. บังคับความกว้างตารางให้ไม่เกินหน้าจอ
            style_table={'overflowX': 'auto', 'width': '100%'},
             
            # 2. บีบขนาดเซลล์ ลดฟอนต์ และลดช่องว่าง (Padding)
            style_cell={
                'textAlign': 'center', 
                'padding': '2px 2px',       # ลดช่องว่างรอบตัวหนังสือให้เหลือน้อยที่สุด
                'fontFamily': 'Arial',
                'fontSize': '11px',         # ลดขนาดฟอนต์ข้อมูลลง
                'minWidth': '40px',         # บีบความกว้างขั้นต่ำ
                'width': '50px',            
                'maxWidth': '65px',         
                'whiteSpace': 'normal'      # ข้อความไหนยาวไป ให้พับขึ้นบรรทัดใหม่
            },
            
            # 3. จัดการหัวตารางให้พับบรรทัดได้
            style_header={
                'backgroundColor': '#2c3e50', 
                'color': 'white', 
                'fontWeight': 'bold',
                'fontSize': '11px',         # ฟอนต์หัวตารางเล็ก
                'height': 'auto',           # ให้ความสูงยืดหยุ่นตามการพับบรรทัด
                'whiteSpace': 'normal'      # ยอมให้คำพับขึ้นบรรทัดใหม่
            },
            
            style_data_conditional=[
                {'if': {'row_index': 'odd'}, 'backgroundColor': '#f9f9f9'},
                {'if': {'state': 'active'}, 'backgroundColor': '#d1ecf1', 'border': '2px solid #17a2b8'} 
            ]
        ),
        html.Hr(style={'margin': '40px 0'}), 
        
        html.Div(id='detail-panel', style={'minHeight': '200px', 'backgroundColor': '#f1f3f5', 'padding': '20px', 'borderRadius': '10px'})
        
    ], style={'paddingTop': '20px'})

def create_realtime_tab():
    
    return html.Div([
    html.H3("Real-Time System Monitoring", style={'textAlign': 'center', 'fontFamily': 'Arial', 'marginBottom': '20px'}),
    # เราจะสร้าง Div เปล่าๆ มารอรับข้อมูลจาก Callback
    html.Div(id='realtime-content', style={'display': 'flex', 'justifyContent': 'center', 'gap': '20px', 'flexWrap': 'wrap'})
], style={'paddingTop': '20px', 'backgroundColor': '#f8f9fa', 'minHeight': '400px'})

# ฟังก์ชันตัวช่วยสำหรับสร้างกล่อง Card สวยๆ (วางไว้ใต้ create_realtime_tab ได้เลย)
def generate_realtime_table(raw_meas, bus_results_dict):
    if not raw_meas or not bus_results_dict:
        return html.H4("กำลังรอข้อมูลจากระบบ...", style={'textAlign': 'center', 'color': '#888'})

    # ฟังก์ชันช่วยคำนวณ Error
    def calc_err(meas, est):
        if meas == 0: return 0
        return round(abs((meas - est) / meas) * 100, 2)

    # 1. จัดเตรียมข้อมูล 6 แถว (P1, Q1, P2, Q2, P3, Q3)
    table_data = [
        {"bus": "", "param": "P1 (W)", "meas": round(raw_meas['bus1']['P'], 2), "est": round(bus_results_dict[1]['p1']['value'], 2), "err": calc_err(raw_meas['bus1']['P'], bus_results_dict[1]['p1']['value'])},
        {"bus": "Bus 1", "param": "Q1 (VAR)", "meas": round(raw_meas['bus1']['Q'], 4), "est": round(bus_results_dict[1]['q1']['value'], 4), "err": calc_err(raw_meas['bus1']['Q'], bus_results_dict[1]['q1']['value'])},
        {"bus": "", "param": "V1 (V)", "meas": round(raw_meas['bus1']['V'], 4), "est": round(bus_results_dict[1]['v1']['value'], 4), "err": calc_err(raw_meas['bus1']['V'], bus_results_dict[1]['v1']['value'])},
        
        {"bus": "", "param": "P2 (W)", "meas": round(raw_meas['bus2']['P'], 2), "est": round(bus_results_dict[2]['p2']['value'], 2), "err": calc_err(raw_meas['bus2']['P'], bus_results_dict[2]['p2']['value'])},
        {"bus": "Bus 2", "param": "Q2 (VAR)", "meas": round(raw_meas['bus2']['Q'], 4), "est": round(bus_results_dict[2]['q2']['value'], 4), "err": calc_err(raw_meas['bus2']['Q'], bus_results_dict[2]['q2']['value'])},
        {"bus": "", "param": "V2 (V)", "meas": round(raw_meas['bus2']['V'], 4), "est": round(bus_results_dict[2]['v2']['value'], 4), "err": calc_err(raw_meas['bus2']['V'], bus_results_dict[2]['v2']['value'])},

        {"bus": "", "param": "P3 (W)", "meas": round(raw_meas['bus3']['P'], 2), "est": round(bus_results_dict[3]['p3']['value'], 2), "err": calc_err(raw_meas['bus3']['P'], bus_results_dict[3]['p3']['value'])},
        {"bus": "Bus 3", "param": "Q3 (VAR)", "meas": round(raw_meas['bus3']['Q'], 4), "est": round(bus_results_dict[3]['q3']['value'], 4), "err": calc_err(raw_meas['bus3']['Q'], bus_results_dict[3]['q3']['value'])},
        {"bus": "", "param": "V3 (V)", "meas": round(raw_meas['bus3']['V'], 4), "est": round(bus_results_dict[3]['v3']['value'], 4), "err": calc_err(raw_meas['bus3']['V'], bus_results_dict[3]['v3']['value'])},
    ]

    # 2. สร้างโครงตาราง Dash Table
    return dash_table.DataTable(
        columns=[
            {"name": "Position", "id": "bus"},
            {"name": "Parameter", "id": "param"},
            {"name": "Meas", "id": "meas"},
            {"name": "Estimate", "id": "est"},
            {"name": "Error (%)", "id": "err"}
        ],
        data=table_data,
        style_table={
            'width': '70%', 
            'marginLeft': '0%',      # ดันจากขอบซ้ายมา 5% (ตารางจะชิดซ้าย)
            # หรือ 'marginLeft': '100px' (ดันเป็นพิกเซลก็ได้)
            'boxShadow': '0 4px 8px rgba(0,0,0,0.1)',
            'if': {'filter_query': '{param} contains "P"'},
                'borderTop': '2px solid #2c3e50'
        },

        style_cell={
            'textAlign': 'center', 'padding': '15px', 
            'fontFamily': 'Arial', 'fontSize': '16px'
        },
        style_header={
            'backgroundColor': '#79ace4', 'color': 'white', 
            'fontWeight': 'bold', 'fontSize': '18px',
            'border': '1px solid #79ace4',       # ลบขอบดำด้านบน/ซ้าย/ขวา
            'borderBottom': '2px solid #4a86e8'  # เพิ่มขอบฟ้าเข้มด้านล่างให้ดูมีมิติ
        },
        
        style_data_conditional=[
            # ทำสีพื้นหลังสลับแถวให้ช่องข้อมูล
            {'if': {'row_index': 'odd'}, 'backgroundColor': '#f8f9fa'},
            
            # บังคับคอลัมน์ Position ให้พื้นหลังขาวล้วนและตัวหนา
            {'if': {'column_id': 'bus'}, 'backgroundColor': 'white', 'fontWeight': 'bold', 'fontSize': '16px'},
            
            # ==========================================
            # 🟢 3. ทริคซ่อนเส้น Merge Cell (ต้องลบทั้งบนและล่างให้ชนกัน) 🟢
            # ==========================================
            # แถว P (บนสุด): ลบเส้นขอบล่างทิ้ง
            {'if': {'column_id': 'bus', 'filter_query': '{param} contains "P"'}, 'borderBottom': 'none'},
            
            # แถว Q (กลาง): ลบทั้งเส้นขอบบน และ เส้นขอบล่างทิ้ง
            {'if': {'column_id': 'bus', 'filter_query': '{param} contains "Q"'}, 'borderTop': 'none', 'borderBottom': 'none'},
            
            # แถว V (ล่างสุด): ลบเส้นขอบบนทิ้ง
            {'if': {'column_id': 'bus', 'filter_query': '{param} contains "V"'}, 'borderTop': 'none'},

            # ==========================================
            # ขีดเส้นทึบแบ่งกลุ่ม Bus (เส้นตีขวางทั้งตาราง)
            # ==========================================
            {'if': {'filter_query': '{param} = "P2 (W)"'}, 'borderTop': '2px solid #adb5bd'},
            {'if': {'filter_query': '{param} = "P3 (W)"'}, 'borderTop': '2px solid #adb5bd'},

            # ไฮไลท์สีแดงเมื่อ Error เกิน 5%
            {'if': {'column_id': 'err', 'filter_query': '{err} > 5'}, 'color': 'red', 'fontWeight': 'bold'},
            
            # ทำคอลัมน์ Parameter เป็นตัวหนา
            {'if': {'column_id': 'param'}, 'fontWeight': 'bold'}
        ]
    )


# Layout
app.layout = html.Div([
    html.H2("3-Bus System (Gen & Load)", style={'textAlign': 'center', 'fontFamily': 'Arial'}),

    html.Div([
        html.Button('START', id='btn-start', n_clicks=0, 
                    style={'backgroundColor': '#28a745', 'color': 'white', 'fontWeight': 'bold', 'padding': '10px 20px', 'marginRight': '10px', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer', 'boxShadow': '0 2px 4px rgba(0,0,0,0.2)'}),
        
        html.Button('STOP', id='btn-stop', n_clicks=0, 
                    style={'backgroundColor': '#dc3545', 'color': 'white', 'fontWeight': 'bold', 'padding': '10px 20px', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer', 'boxShadow': '0 2px 4px rgba(0,0,0,0.2)'})
    ], style={'textAlign': 'center', 'marginBottom': '20px'}),
    
    dcc.Tabs(id="tabs-menu", value='tab-sld', children=[
        dcc.Tab(label='Single-Line Diagram', value='tab-sld', children=create_sld_tab()),
        dcc.Tab(label='Real-Time Dashboard', value='tab-realtime', children=create_realtime_tab()),
        dcc.Tab(label='Data Report', value='tab-data', children=create_report_tab()),
    ]),

    
        
    dcc.Interval(id='interval-update', interval=2000, n_intervals=0, disabled=True)  # เริ่มต้นให้ Interval หยุดทำงาน
])


def update_history_table(raw_meas, bus_results_dict, history_list):
    """นำค่าดิบ (Meas) และค่าประมาณ (Est) มาสร้างเป็น 1 แถวประวัติ (Time Series)"""
    now = datetime.now().strftime("%H:%M:%S")
    
    # ดึงค่า Error (ถ้าอยากคำนวณ) หรือใส่ 0 ไว้ก่อน
    def calc_err(meas, est):
        if meas == 0: return 0
        return round(abs((meas - est) / meas) * 100, 2)

    new_row = {
        "time": now,
        # -------- BUS 1 --------
        "p1_meas": round(raw_meas['bus1']['P'], 2),
        "p1_est": round(bus_results_dict[1]['p1']['value'], 2),
        "p1_err": calc_err(raw_meas['bus1']['P'], bus_results_dict[1]['p1']['value']),
        "q1_meas": round(raw_meas['bus1']['Q'], 2),
        "q1_est": round(bus_results_dict[1]['q1']['value'], 2),
        "q1_err": calc_err(raw_meas['bus1']['Q'], bus_results_dict[1]['q1']['value']),
        
        # -------- BUS 2 --------
        "p2_meas": round(raw_meas['bus2']['P'], 2),
        "p2_est": round(bus_results_dict[2]['p2']['value'], 2),
        "p2_err": calc_err(raw_meas['bus2']['P'], bus_results_dict[2]['p2']['value']),
        "q2_meas": round(raw_meas['bus2']['Q'], 2),
        "q2_est": round(bus_results_dict[2]['q2']['value'], 2),
        "q2_err": calc_err(raw_meas['bus2']['Q'], bus_results_dict[2]['q2']['value']),

        # -------- BUS 3 --------
        "p3_meas": round(raw_meas['bus3']['P'], 2),
        "p3_est": round(bus_results_dict[3]['p3']['value'], 2),
        "p3_err": calc_err(raw_meas['bus3']['P'], bus_results_dict[3]['p3']['value']),
        "q3_meas": round(raw_meas['bus3']['Q'], 2),
        "q3_est": round(bus_results_dict[3]['q3']['value'], 2),
        "q3_err": calc_err(raw_meas['bus3']['Q'], bus_results_dict[3]['q3']['value']),
    }
    
    history_list.append(new_row)
    # ตัดให้แสดงแค่ 50 แถวล่าสุด ป้องกันเว็บค้าง
    if len(history_list) > 50:
        history_list.pop(0)
        
    return history_list



@app.callback(
    Output('sld-graph', 'figure'),
    Output('data-table', 'data'),
    Output('realtime-content', 'children'),
    Input('interval-update', 'n_intervals')
)
def update_sld(n):
    
    global history_data  

    if n is None:
        n = 0
    

    # meas_val = group.get_measurements()

    raw_meas = {
        'bus1': {'P':217.84, 
                 'Q':58.784, 
                 'V': 368.54},
        'bus2': {'P':138.18, 
                 'Q':0.1359, 
                 'V': 362.3988},
        'bus3': {'P':77.79, 
                 'Q':51.2691, 
                 'V': 361.4268}
    }

    
    
    # print("\nRaw measurements from Modbus:", meas_val)
        # meas_val = converter.to_pu_batch(meas_val)
        # test
    grid.add_measurement(position = 'bus', name="p1", id=1,  pos_id=1, mvalue=raw_meas['bus1']['P'], msd=0.010)
    grid.add_measurement(position = 'bus', name="q1", id=2,  pos_id=1, mvalue=raw_meas['bus1']['Q'], msd=0.010)
    grid.add_measurement(position = 'bus', name="p2", id=3,  pos_id=2, mvalue=raw_meas['bus2']['P'], msd=0.010)
    grid.add_measurement(position = 'bus', name="q2", id=4,  pos_id=2, mvalue=raw_meas['bus2']['Q'], msd=0.010)
    grid.add_measurement(position = 'bus', name="p3", id=5,  pos_id=3, mvalue=raw_meas['bus3']['P'], msd=0.010)
    grid.add_measurement(position = 'bus', name="q3", id=6,  pos_id=3, mvalue=raw_meas['bus3']['Q'], msd=0.010)
    grid.add_measurement(position = 'bus', name="v1", id=7,  pos_id=1, mvalue=raw_meas['bus1']['V'], msd=0.010)
    grid.add_measurement(position = 'bus', name="v2", id=8,  pos_id=2, mvalue=raw_meas['bus2']['V'], msd=0.010)
    grid.add_measurement(position = 'bus', name="v3", id=9,  pos_id=3, mvalue=raw_meas['bus3']['V'], msd=0.010)
    
    grid.build_system()    # get_ready_measurements() runs inside here
    grid.estimate()
    # 4. ดึงค่า (Get Results) ออกมาจาก Grid!
    
    raw_bus_dict = grid.get_bus_results()
    
    raw_branch_dict = grid.get_branch_results()
    print("\nBranch Results:", raw_branch_dict)
    
    meas = grid.get_measurement_var()

    meas_var = {
        'bus1': {'P':meas[1]['p1']['value'], 
                 'Q':meas[1]['q1']['value'], 
                 'V': meas[1]['v1']['value']
                 },
        'bus2': {'P':meas[2]['p2']['value'], 
                 'Q':meas[2]['q2']['value'], 
                 'V': meas[2]['v2']['value']
                 },
        'bus3': {'P':meas[3]['p3']['value'], 
                 'Q':meas[3]['q3']['value'], 
                 'V': meas[3]['v3']['value']
                 }
        }

    # bus_dict = {
    #     'bus1': {'P': raw_bus_dict[1]['p1']['value'], 'Q': raw_bus_dict[1]['q1']['value'], 'V': raw_bus_dict[1]['vmag']['value']},
    #     'bus2': {'P': raw_bus_dict[2]['p2']['value'], 'Q': raw_bus_dict[2]['q2']['value'], 'V': raw_bus_dict[2]['vmag']['value']},
    #     'bus3': {'P': raw_bus_dict[3]['p3']['value'], 'Q': raw_bus_dict[3]['q3']['value'], 'V': raw_bus_dict[3]['vmag']['value']}
    # }
    # 5. อัปเดตข้อมูลตาราง
    history_data = update_history_table(meas_var, raw_bus_dict, history_data)

    # 6. วาดรูปกราฟ SLD (คุณต้องปรับฟังก์ชัน create_sld_figure ให้รับ raw_bus_dict)
    fig = create_sld_figure(raw_bus_dict, raw_branch_dict, meas)
    realtime_table_html = generate_realtime_table(meas_var, raw_bus_dict)

    return fig, history_data,realtime_table_html

@app.callback(
    Output('detail-panel', 'children'),
    Input('data-table', 'active_cell'),
    State('data-table', 'data')
)
def display_detail_panel(active_cell, table_data):
    if active_cell is None or table_data is None or len(table_data) == 0:
        return create_detail_cards(None)
    
    row_index = active_cell['row']
    selected_row_data = table_data[row_index]
    
    return create_detail_cards(selected_row_data)


@app.callback(
    Output('interval-update', 'disabled'),
    Input('btn-start', 'n_clicks'),
    Input('btn-stop', 'n_clicks'),
    State('interval-update', 'disabled')
)
def control_interval(start_clicks, stop_clicks, is_disabled):
    # เช็คว่าปุ่มไหนเพิ่งถูกกด
    trigger = ctx.triggered_id
    
    if trigger == 'btn-stop':
        return True   # สั่งปิดการทำงาน (Interval หยุด)
    elif trigger == 'btn-start':
        return False  # สั่งเปิดการทำงาน (Interval เดินต่อ)
        
    return is_disabled # ถ้ายังไม่กดอะไร ให้คงสถานะเดิมไว้

# 1. สร้างฟังก์ชันสำหรับรันเซิร์ฟเวอร์ Dash ให้ทำงานอยู่เบื้องหลัง (Background)
def run_dash_server():
    # บังคับปิด debug และ reloader เพื่อไม่ให้มีปัญหากับหน้าต่างแอป
    app.run(port=8050, debug=False, use_reloader=False)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    # 2. แยก Thread ให้ Dash รันคู่ขนานไปกับหน้าต่างโปรแกรม
    t = Thread(target=run_dash_server)
    t.daemon = True # สั่งให้เซิร์ฟเวอร์ปิดตัวเองอัตโนมัติเมื่อเรากากบาทปิดหน้าต่างโปรแกรม
    t.start()
    
    # 3. สร้างหน้าต่าง Desktop และดึงเว็บ Dash มาแสดงผล
    webview.create_window(
        title='Mahidol Power System State Estimation', # ชื่อบนหัวหน้าต่างโปรแกรม
        url='http://127.0.0.1:8050/', 
        width=1280,   # ความกว้างหน้าต่าง
        height=800,   # ความสูงหน้าต่าง
        min_size=(1024, 768) # บังคับขนาดเล็กสุดไม่ให้คนหดหน้าต่างจนกราฟพัง
    )
    
    # 4. สั่งเปิดหน้าต่างโปรแกรมขึ้นมาเลย!
    webview.start()
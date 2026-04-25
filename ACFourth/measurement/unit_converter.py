# measurement/unit_converter.py
import dataclasses
import numpy as np
from typing import Optional, Any, List

# สมมติว่าดึงมาจากไฟล์ structural_data.py ของคุณ
from structural_data import Load, Generator, Measurement

PREFIX_MULTIPLIERS = {
    'T': 1e12, 'tera': 1e12,
    'G': 1e9,  'giga': 1e9,
    'M': 1e6,  'mega': 1e6,
    'k': 1e3,  'kilo': 1e3,
    'h': 1e2,  'hecto': 1e2,
    'da': 1e1, 'deca': 1e1,
    '': 1.0,   'base': 1.0, 'none': 1.0,
    'd': 1e-1, 'deci': 1e-1,
    'c': 1e-2, 'centi': 1e-2,
    'm': 1e-3, 'milli': 1e-3,
    'u': 1e-6, 'µ': 1e-6, 'micro': 1e-6,
    'n': 1e-9, 'nano': 1e-9,
    'p': 1e-12, 'pico': 1e-12,
}

class PrefixConverter:
    """แปลงค่าทางไฟฟ้าที่มีหน่วย Prefix ให้เป็นหน่วยฐาน"""
    @staticmethod
    def _get_multiplier(unit: str) -> float:
        if not unit: return 1.0
        unit_stripped, unit_lower = unit.strip(), unit.strip().lower()
        prefixes = sorted(PREFIX_MULTIPLIERS.keys(), key=len, reverse=True)
        
        for prefix in prefixes:
            if not prefix: continue
            if len(prefix) > 1:
                if unit_lower.startswith(prefix.lower()): return PREFIX_MULTIPLIERS[prefix]
            else:
                if unit_stripped.startswith(prefix): return PREFIX_MULTIPLIERS[prefix]
        return 1.0

    @staticmethod
    def _get_keys(obj: Any) -> List[str]:
        if isinstance(obj, dict): return list(obj.keys())
        elif dataclasses.is_dataclass(obj): return [f.name for f in dataclasses.fields(obj)]
        elif hasattr(obj, '__dict__'): return list(obj.__dict__.keys())
        return []

    @staticmethod
    def to_base(obj: Any, target_fields: Optional[List[str]] = None) -> Any:
        is_dict = isinstance(obj, dict)
        unit_str = obj.get('unit') if is_dict else getattr(obj, 'unit', None)
        if not unit_str: return obj
            
        multiplier = PrefixConverter._get_multiplier(unit_str)
        if multiplier == 1.0: return obj

        keys = target_fields if target_fields else PrefixConverter._get_keys(obj)
        for key in keys:
            if key in ('id', 'pos_id', 'bus_id', 'branch_id', 'unit') or key.endswith('_pu'):
                continue
            val = obj.get(key) if is_dict else getattr(obj, key, None)
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                is_var = 'var' in key.lower() and key.lower() != 'var_pu'
                actual_multiplier = multiplier ** 2 if is_var else multiplier
                new_val = float(val) * actual_multiplier
                if is_dict: obj[key] = new_val
                else: setattr(obj, key, new_val)
        return obj

    @staticmethod
    def from_base(obj: Any, target_unit: str, target_fields: Optional[List[str]] = None) -> Any:
        """
        แปลงค่าจากหน่วยฐานเป็นหน่วย Prefix ตามที่ระบุ (เช่น target_unit="kilo") (In-place)
        """
        is_dict = isinstance(obj, dict)
        multiplier = PrefixConverter._get_multiplier(target_unit)
        
        if multiplier == 1.0:
            return obj

        keys = target_fields if target_fields else PrefixConverter._get_keys(obj)

        for key in keys:
            if key in ('id', 'pos_id', 'bus_id', 'branch_id', 'unit') or key.endswith('_pu'):
                continue
                
            val = obj.get(key) if is_dict else getattr(obj, key, None)
            
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                is_var = 'var' in key.lower() and key.lower() != 'var_pu'
                
                actual_multiplier = multiplier ** 2 if is_var else multiplier
                new_val = float(val) / actual_multiplier # หารเพื่อแปลงเป็นหน่วยที่ใหญ่ขึ้น
                
                if is_dict:
                    obj[key] = new_val
                else:
                    setattr(obj, key, new_val)
                    
        # อัปเดตชื่อหน่วยใหม่ให้ Object
        if is_dict:
            obj['unit'] = target_unit
        else:
            if hasattr(obj, 'unit'):
                setattr(obj, 'unit', target_unit)

        return obj

    @staticmethod
    def to_base_batch(objs: List[Any], target_fields: Optional[List[str]] = None) -> List[Any]:
        for obj in objs:
            PrefixConverter.to_base(obj, target_fields)
        return objs

    @staticmethod
    def from_base_batch(objs: List[Any], target_unit: str, target_fields: Optional[List[str]] = None) -> List[Any]:
        for obj in objs:
            PrefixConverter.from_base(obj, target_unit, target_fields)
        return objs


class PerUnitConvert:
    """
    Utility Class สำหรับแปลงค่าทางไฟฟ้าให้อยู่ในระบบ Per Unit (pu) 
    และแปลงกลับเป็นค่าจริง (Physical Value) (ใช้งานแบบ Static Methods)
    """

    @staticmethod
    def _get_base(mtype: str, V_base: float, S_base: float) -> Optional[float]:
        """เลือก Base ที่เหมาะสมโดยดูจากอักษรตัวแรกของชนิดตัวแปร"""
        if not mtype:
            return None
        mtype = mtype.lower()
        if mtype.startswith("v") and (not mtype.endswith("deg") and not mtype.endswith("angle")): 
            return V_base
        if mtype.startswith("i"): 
            return S_base / (np.sqrt(3) * V_base) # คำนวณ I_base สดๆ
        if mtype.startswith(("p", "q", "s")): 
            return S_base
        return None

    @staticmethod
    def _to_pu_scalar(mtype: str, value: float, V_base: float, S_base: float, is_variance: bool = False) -> float:
        """คำนวณค่าจริง -> Per Unit"""
        base = PerUnitConvert._get_base(mtype, V_base, S_base)
        if base is None or base == 0:
            return value
        
        if is_variance:
            return value / (base ** 2)
        return value / base

    @staticmethod
    def _from_pu_scalar(mtype: str, value_pu: float, V_base: float, S_base: float, is_variance: bool = False) -> float:
        """คำนวณ Per Unit -> ค่าจริง"""
        base = PerUnitConvert._get_base(mtype, V_base, S_base)
        if base is None:
            return value_pu
            
        if is_variance:
            return value_pu * (base ** 2)
        return value_pu * base

    @staticmethod
    def to_pu(obj: Any, V_base: float = 380.0, S_base: float = 1000.0) -> Any:
        """แปลงค่าใน Data Class เป็น Per Unit (In-place)"""
        if not dataclasses.is_dataclass(obj):
            return obj

        main_mtype = getattr(obj, 'mtype', None)

        for field in dataclasses.fields(obj):
            attr_name = field.name
            
            if attr_name.endswith('_pu'):
                base_attr = attr_name[:-3] 
                
                if hasattr(obj, base_attr):
                    val_pu = getattr(obj, attr_name)
                    val = getattr(obj, base_attr)

                    if val_pu is None and val is not None:
                        mtype_to_use = main_mtype if main_mtype else base_attr 
                        is_var = 'var' in base_attr.lower()
                        
                        # ส่ง V_base และ S_base ลงไปใน helper
                        new_pu_val = PerUnitConvert._to_pu_scalar(
                            mtype_to_use, val, V_base, S_base, is_variance=is_var
                        )
                        setattr(obj, attr_name, new_pu_val)
        return obj
        
    @staticmethod
    def from_pu(obj: Any, V_base: float = 380.0, S_base: float = 1000.0) -> Any:
        """แปลงค่าใน Data Class จาก Per Unit กลับเป็นค่าจริง (In-place)"""
        if not dataclasses.is_dataclass(obj):
            return obj

        main_mtype = getattr(obj, 'mtype', None)

        for field in dataclasses.fields(obj):
            attr_name = field.name
            
            if attr_name.endswith('_pu'):
                base_attr = attr_name[:-3] 
                
                if hasattr(obj, base_attr):
                    val_pu = getattr(obj, attr_name)
                    val = getattr(obj, base_attr)

                    if val is None and val_pu is not None:
                        mtype_to_use = main_mtype if main_mtype else base_attr 
                        is_var = 'var' in base_attr.lower()
                        
                        new_val = PerUnitConvert._from_pu_scalar(
                            mtype_to_use, val_pu, V_base, S_base, is_variance=is_var
                        )
                        setattr(obj, base_attr, new_val)
        return obj

    @staticmethod
    def to_pu_batch(objs: List[Any], V_base: float = 380.0, S_base: float = 1000.0) -> List[Any]:
        """รันแปลงค่าเป็น Per Unit ให้กับ List"""
        for obj in objs:
            PerUnitConvert.to_pu(obj, V_base, S_base)
        return objs

    @staticmethod
    def from_pu_batch(objs: List[Any], V_base: float = 380.0, S_base: float = 1000.0) -> List[Any]:
        """รันแปลงค่าจาก Per Unit กลับเป็นค่าจริง ให้กับ List"""
        for obj in objs:
            PerUnitConvert.from_pu(obj, V_base, S_base)
        return objs
    

class PowerDirection:
    @staticmethod
    def adjust_load_values(measurements: List['Measurement'], loads: List['Load']):
        load_buses = {load.lbus for load in loads}
        for m in measurements:
            if m.pos_id in load_buses and m.mtype and m.mtype.lower() in ['pinject', 'qinject']:
                m.mvalue = -abs(m.mvalue)
                if m.mvalue_pu is not None:
                    m.mvalue_pu = -abs(m.mvalue_pu)

    @staticmethod
    def adjust_gen_values(measurements: List['Measurement'], generators: List['Generator']):
        gen_buses = {gen.gbus for gen in generators}
        for m in measurements:
            if m.pos_id in gen_buses and m.mtype and m.mtype.lower() in ['pinject', 'qinject']:
                m.mvalue = abs(m.mvalue)
                if m.mvalue_pu is not None:
                    m.mvalue_pu = abs(m.mvalue_pu)
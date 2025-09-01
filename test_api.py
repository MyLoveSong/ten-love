import requests
import json
import time

def test_api():
    """测试API功能"""
    
    # API基础URL
    base_url = "http://localhost:8000"
    
    # 测试健康检查
    print("=== 测试健康检查 ===")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"健康检查结果: {response.json()}")
    except Exception as e:
        print(f"健康检查失败: {e}")
        return
    
    # 测试预测接口
    print("\n=== 测试血糖预测 ===")
    
    # 示例数据
    test_data = {
        "age": 45.0,
        "bmi": 25.5,
        "blood_pressure": 130.0,
        "fasting_glucose": 6.8,
        "postprandial_glucose": 8.5,
        "hba1c": 6.2,
        "insulin": 15.0,
        "cholesterol": 200.0,
        "ldl": 120.0,
        "hdl": 50.0,
        "triglycerides": 150.0,
        "physical_activity": 3,
        "sleep_quality": 4,
        "stress_level": 2,
        "gender": 1,
        "diabetes_type": 1,
        "pregnant": 0
    }
    
    try:
        response = requests.post(f"{base_url}/predict/glucose", json=test_data)
        if response.status_code == 200:
            result = response.json()
            print("预测成功！")
            print(f"预测血糖: {result['predicted_glucose']:.2f} mmol/L")
            print(f"风险等级: {result['risk_level']}")
            print(f"置信度: {result['confidence']:.2f}")
            print(f"建议: {', '.join(result['recommendations'])}")
            print(f"\nAI健康建议:\n{result['health_advice']}")
        else:
            print(f"预测失败: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"API调用失败: {e}")
    
    # 测试食物推荐接口
    print("\n=== 测试食物推荐 ===")
    try:
        params = {
            "glucose_level": 7.5,
            "cultural_preferences": ["清真", "川菜"]
        }
        response = requests.post(f"{base_url}/recommend/food", json=params)
        if response.status_code == 200:
            result = response.json()
            print("推荐成功！")
            print(f"早餐推荐: {', '.join(result['recommendations']['breakfast'])}")
            print(f"午餐推荐: {', '.join(result['recommendations']['lunch'])}")
            print(f"晚餐推荐: {', '.join(result['recommendations']['dinner'])}")
            if 'avoid' in result['recommendations'] and result['recommendations']['avoid']:
                print(f"建议避免: {', '.join(result['recommendations']['avoid'])}")
        else:
            print(f"推荐失败: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"API调用失败: {e}")

if __name__ == "__main__":
    print("开始测试API...")
    print("等待API服务启动...")
    time.sleep(2)  # 给API服务一些启动时间
    test_api() 
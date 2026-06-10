"""
知识图谱适配器 - 集成OHDSI-Achilles和Diabetes-KG-CN
用于提升血糖预测模型的精度和临床决策支持能力
"""

import requests
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

# 尝试导入Neo4j相关库
try:
    from py2neo import Graph, Node, Relationship
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    logging.warning("Neo4j not available. Install py2neo for full functionality.")

logger = logging.getLogger(__name__)

@dataclass
class KGConfig:
    """知识图谱配置"""
    # OHDSI-Achilles配置
    ohdsi_base_url: str = "https://fhir.ohdsi.org/api"
    ohdsi_api_key: Optional[str] = None
    ohdsi_timeout: int = 30

    # Diabetes-KG-CN配置
    kg_cn_url: str = "http://kg-diabetes.cn:7474"
    kg_cn_username: str = "neo4j"
    kg_cn_password: str = "password"

    # 瑞金糖尿病知识图谱配置
    ruijin_kg_path: Optional[str] = None
    ruijin_api_key: Optional[str] = None

    # DiabRe-KG配置
    diabre_kg_path: Optional[str] = None

    # OpenFDA配置
    openfda_base_url: str = "https://api.fda.gov"
    openfda_api_key: Optional[str] = None
    openfda_timeout: int = 30

    # ClinicalTrials.gov配置
    clinicaltrials_base_url: str = "https://clinicaltrials.gov/api/v2"
    clinicaltrials_timeout: int = 30

    # USDA FoodData Central配置
    usda_base_url: str = "https://api.nal.usda.gov/fdc/v1"
    usda_api_key: Optional[str] = None
    usda_timeout: int = 30

    # WHO Global Health Observatory配置
    who_base_url: str = "https://apps.who.int/gho/athena/api"
    who_timeout: int = 30

    # 缓存配置
    cache_enabled: bool = True
    cache_ttl_hours: int = 24

@dataclass
class DataSourceQuality:
    """数据源质量评估"""
    source_name: str
    relevance_score: float  # 相关性评分 (0-1)
    completeness_score: float  # 完整性评分 (0-1)
    accuracy_score: float  # 准确性评分 (0-1)
    timeliness_score: float  # 时效性评分 (0-1)
    overall_quality: float  # 综合质量评分 (0-1)

    def __post_init__(self):
        # 计算综合质量评分
        self.overall_quality = (
            self.relevance_score * 0.4 +
            self.completeness_score * 0.3 +
            self.accuracy_score * 0.2 +
            self.timeliness_score * 0.1
        )

@dataclass
class PatientKGFeatures:
    """患者知识图谱特征"""
    # 胰岛素-碳水化合物比值
    icr_ratio: Optional[float] = None

    # 地域饮食偏好
    regional_gi_preferences: Dict[str, float] = None

    # 并发症风险因子
    complication_risk_factors: List[str] = None

    # 药物相互作用
    drug_interactions: List[Dict[str, Any]] = None

    # 饮食建议
    dietary_recommendations: List[str] = None

    # 运动建议
    exercise_recommendations: List[str] = None

    # 特征质量评分
    feature_quality_scores: Dict[str, float] = None

    def __post_init__(self):
        if self.regional_gi_preferences is None:
            self.regional_gi_preferences = {}
        if self.complication_risk_factors is None:
            self.complication_risk_factors = []
        if self.drug_interactions is None:
            self.drug_interactions = []
        if self.dietary_recommendations is None:
            self.dietary_recommendations = []
        if self.exercise_recommendations is None:
            self.exercise_recommendations = []
        if self.feature_quality_scores is None:
            self.feature_quality_scores = {}

class BaseKGAdapter(ABC):
    """知识图谱适配器基类"""

    def __init__(self, config: KGConfig):
        self.config = config
        self.cache = {} if config.cache_enabled else None

    @abstractmethod
    def get_patient_features(self, patient_id: str, patient_info: Dict[str, Any]) -> PatientKGFeatures:
        """获取患者知识图谱特征"""
        pass

    def _get_cache_key(self, patient_id: str, feature_type: str) -> str:
        """生成缓存键"""
        return f"{patient_id}_{feature_type}"

    def _is_cache_valid(self, cache_key: str) -> bool:
        """检查缓存是否有效"""
        if not self.cache or cache_key not in self.cache:
            return False

        cache_time = self.cache[cache_key].get('timestamp')
        if not cache_time:
            return False

        return datetime.now() - cache_time < timedelta(hours=self.config.cache_ttl_hours)

    def _set_cache(self, cache_key: str, data: Any):
        """设置缓存"""
        if self.cache:
            self.cache[cache_key] = {
                'data': data,
                'timestamp': datetime.now()
            }

    def _get_cache(self, cache_key: str) -> Any:
        """获取缓存"""
        if self.cache and self._is_cache_valid(cache_key):
            return self.cache[cache_key]['data']
        return None

class OHSDIAdapter(BaseKGAdapter):
    """OHDSI-Achilles适配器"""

    def __init__(self, config: KGConfig):
        super().__init__(config)
        self.session = requests.Session()
        if config.ohdsi_api_key:
            self.session.headers.update({'Authorization': f'Bearer {config.ohdsi_api_key}'})

    def get_patient_features(self, patient_id: str, patient_info: Dict[str, Any]) -> PatientKGFeatures:
        """获取患者OHDSI特征"""
        features = PatientKGFeatures()

        try:
            # 获取胰岛素-碳水化合物比值
            features.icr_ratio = self._get_icr_ratio(patient_id)

            # 获取并发症风险因子
            features.complication_risk_factors = self._get_complication_risk_factors(patient_id)

            # 获取药物相互作用
            features.drug_interactions = self._get_drug_interactions(patient_id)

            logger.info(f"Successfully retrieved OHDSI features for patient {patient_id}")

        except Exception as e:
            logger.error(f"Error retrieving OHDSI features for patient {patient_id}: {e}")

        return features

    def _get_icr_ratio(self, patient_id: str) -> Optional[float]:
        """获取胰岛素-碳水化合物比值"""
        cache_key = self._get_cache_key(patient_id, "icr")
        cached_data = self._get_cache(cache_key)
        if cached_data is not None:
            return cached_data

        try:
            url = f"{self.config.ohdsi_base_url}/DiabetesKG/Observation"
            params = {
                'patient': patient_id,
                'code': '33747-0'  # 胰岛素-碳水化合物比值代码
            }

            response = self.session.get(url, params=params, timeout=self.config.ohdsi_timeout)
            response.raise_for_status()

            data = response.json()
            if 'valueQuantity' in data and 'value' in data['valueQuantity']:
                icr_value = float(data['valueQuantity']['value'])
                self._set_cache(cache_key, icr_value)
                return icr_value

        except Exception as e:
            logger.warning(f"Failed to get ICR ratio for patient {patient_id}: {e}")

        return None

    def _get_complication_risk_factors(self, patient_id: str) -> List[str]:
        """获取并发症风险因子"""
        cache_key = self._get_cache_key(patient_id, "complication_risk")
        cached_data = self._get_cache(cache_key)
        if cached_data is not None:
            return cached_data

        risk_factors = []

        try:
            # 获取糖尿病并发症相关观察
            url = f"{self.config.ohdsi_base_url}/DiabetesKG/Observation"
            params = {
                'patient': patient_id,
                'category': 'diabetes-complications'
            }

            response = self.session.get(url, params=params, timeout=self.config.ohdsi_timeout)
            response.raise_for_status()

            data = response.json()
            if 'entry' in data:
                for entry in data['entry']:
                    if 'resource' in entry and 'code' in entry['resource']:
                        code = entry['resource']['code']
                        if 'coding' in code:
                            for coding in code['coding']:
                                if 'display' in coding:
                                    risk_factors.append(coding['display'])

            self._set_cache(cache_key, risk_factors)

        except Exception as e:
            logger.warning(f"Failed to get complication risk factors for patient {patient_id}: {e}")

        return risk_factors

    def _get_drug_interactions(self, patient_id: str) -> List[Dict[str, Any]]:
        """获取药物相互作用"""
        cache_key = self._get_cache_key(patient_id, "drug_interactions")
        cached_data = self._get_cache(cache_key)
        if cached_data is not None:
            return cached_data

        interactions = []

        try:
            # 获取药物相互作用
            url = f"{self.config.ohdsi_base_url}/DiabetesKG/MedicationStatement"
            params = {'patient': patient_id}

            response = self.session.get(url, params=params, timeout=self.config.ohdsi_timeout)
            response.raise_for_status()

            data = response.json()
            if 'entry' in data:
                for entry in data['entry']:
                    if 'resource' in entry:
                        medication = entry['resource']
                        if 'medicationCodeableConcept' in medication:
                            drug_info = {
                                'name': medication['medicationCodeableConcept'].get('text', 'Unknown'),
                                'interactions': []  # 这里可以进一步查询相互作用
                            }
                            interactions.append(drug_info)

            self._set_cache(cache_key, interactions)

        except Exception as e:
            logger.warning(f"Failed to get drug interactions for patient {patient_id}: {e}")

        return interactions

class DiabetesKGCNAdapter(BaseKGAdapter):
    """Diabetes-KG-CN适配器"""

    def __init__(self, config: KGConfig):
        super().__init__(config)
        self.graph = None
        if NEO4J_AVAILABLE:
            try:
                self.graph = Graph(
                    config.kg_cn_url,
                    auth=(config.kg_cn_username, config.kg_cn_password)
                )
                logger.info("Successfully connected to Diabetes-KG-CN")
            except Exception as e:
                logger.warning(f"Failed to connect to Diabetes-KG-CN: {e}")

    def get_patient_features(self, patient_id: str, patient_info: Dict[str, Any]) -> PatientKGFeatures:
        """获取患者Diabetes-KG-CN特征"""
        features = PatientKGFeatures()

        if not self.graph:
            logger.warning("Neo4j graph not available")
            return features

        try:
            # 获取地域饮食偏好
            region = patient_info.get('region', '全国')
            features.regional_gi_preferences = self._get_regional_gi_preferences(region)

            # 获取饮食建议
            features.dietary_recommendations = self._get_dietary_recommendations(patient_info)

            # 获取运动建议
            features.exercise_recommendations = self._get_exercise_recommendations(patient_info)

            logger.info(f"Successfully retrieved Diabetes-KG-CN features for patient {patient_id}")

        except Exception as e:
            logger.error(f"Error retrieving Diabetes-KG-CN features for patient {patient_id}: {e}")

        return features

    def _get_regional_gi_preferences(self, region: str) -> Dict[str, float]:
        """获取地域饮食偏好和GI值"""
        cache_key = self._get_cache_key(region, "regional_gi")
        cached_data = self._get_cache(cache_key)
        if cached_data is not None:
            return cached_data

        gi_preferences = {}

        try:
            # 查询地域饮食偏好
            cypher_query = """
            MATCH (r:Region {name: $region})-[:PREFERS]->(f:Food)
            RETURN f.name as food_name, f.GI as gi_value
            LIMIT 20
            """

            result = self.graph.run(cypher_query, region=region)

            for record in result:
                food_name = record['food_name']
                gi_value = record['gi_value']
                if gi_value:
                    gi_preferences[food_name] = float(gi_value)

            self._set_cache(cache_key, gi_preferences)

        except Exception as e:
            logger.warning(f"Failed to get regional GI preferences for region {region}: {e}")

        return gi_preferences

    def _get_dietary_recommendations(self, patient_info: Dict[str, Any]) -> List[str]:
        """获取饮食建议"""
        recommendations = []

        try:
            diabetes_type = patient_info.get('diabetes_type', 'type2')
            age = patient_info.get('age', 50)
            bmi = patient_info.get('bmi', 25)

            # 基于糖尿病类型和患者特征的饮食建议
            cypher_query = """
            MATCH (dt:DiabetesType {name: $diabetes_type})-[:RECOMMENDS]->(dr:DietaryRecommendation)
            WHERE dr.min_age <= $age AND dr.max_age >= $age
            AND dr.min_bmi <= $bmi AND dr.max_bmi >= $bmi
            RETURN dr.recommendation as recommendation
            ORDER BY dr.priority DESC
            LIMIT 5
            """

            result = self.graph.run(cypher_query,
                                  diabetes_type=diabetes_type,
                                  age=age,
                                  bmi=bmi)

            for record in result:
                recommendations.append(record['recommendation'])

        except Exception as e:
            logger.warning(f"Failed to get dietary recommendations: {e}")

        return recommendations

    def _get_exercise_recommendations(self, patient_info: Dict[str, Any]) -> List[str]:
        """获取运动建议"""
        recommendations = []

        try:
            diabetes_type = patient_info.get('diabetes_type', 'type2')
            age = patient_info.get('age', 50)

            # 基于糖尿病类型和年龄的运动建议
            cypher_query = """
            MATCH (dt:DiabetesType {name: $diabetes_type})-[:RECOMMENDS]->(er:ExerciseRecommendation)
            WHERE er.min_age <= $age AND er.max_age >= $age
            RETURN er.recommendation as recommendation
            ORDER BY er.priority DESC
            LIMIT 3
            """

            result = self.graph.run(cypher_query,
                                  diabetes_type=diabetes_type,
                                  age=age)

            for record in result:
                recommendations.append(record['recommendation'])

        except Exception as e:
            logger.warning(f"Failed to get exercise recommendations: {e}")

        return recommendations

class RuijinKGAdapter(BaseKGAdapter):
    """瑞金糖尿病知识图谱适配器"""

    def __init__(self, config: KGConfig):
        super().__init__(config)
        self.kg_data = None
        self.api_base_url = "https://api.ruijin-diabetes-kg.com"  # 模拟API地址
        if config.ruijin_kg_path:
            self._load_kg_data()

    def _load_kg_data(self):
        """加载瑞金糖尿病知识图谱数据"""
        try:
            if self.config.ruijin_kg_path.endswith('.csv'):
                self.kg_data = pd.read_csv(self.config.ruijin_kg_path)
            elif self.config.ruijin_kg_path.endswith('.json'):
                with open(self.config.ruijin_kg_path, 'r', encoding='utf-8') as f:
                    self.kg_data = json.load(f)
            logger.info("Successfully loaded Ruijin Diabetes KG data")
        except Exception as e:
            logger.warning(f"Failed to load Ruijin Diabetes KG data: {e}")

    def get_patient_features(self, patient_id: str, patient_info: Dict[str, Any]) -> PatientKGFeatures:
        """获取患者瑞金糖尿病知识图谱特征"""
        features = PatientKGFeatures()

        try:
            # 获取临床指南建议
            features.dietary_recommendations.extend(
                self._get_clinical_guidelines(patient_info)
            )

            # 获取文献支持的治疗建议
            features.drug_interactions.extend(
                self._get_evidence_based_treatments(patient_info)
            )

            # 获取并发症预防建议
            features.complication_risk_factors.extend(
                self._get_complication_prevention(patient_info)
            )

        except Exception as e:
            logger.error(f"Error retrieving Ruijin KG features for patient {patient_id}: {e}")

        return features

    def _get_clinical_guidelines(self, patient_info: Dict[str, Any]) -> List[str]:
        """获取临床指南建议"""
        recommendations = []

        try:
            diabetes_type = patient_info.get('diabetes_type', 'type2')
            age = patient_info.get('age', 50)
            hba1c = patient_info.get('hba1c', 7.0)

            # 基于糖尿病类型和HbA1c的指南建议
            if diabetes_type == 'type1':
                if hba1c > 7.5:
                    recommendations.append("根据ADA指南，建议强化胰岛素治疗方案")
                    recommendations.append("考虑使用连续血糖监测(CGM)优化血糖控制")
                else:
                    recommendations.append("当前血糖控制良好，维持现有治疗方案")
            elif diabetes_type == 'type2':
                if hba1c > 8.0:
                    recommendations.append("根据ADA指南，建议起始或强化药物治疗")
                    recommendations.append("优先考虑二甲双胍作为一线治疗")
                else:
                    recommendations.append("血糖控制达标，继续生活方式干预")
            elif diabetes_type == 'gdm':
                recommendations.append("根据ACOG指南，严格控制餐后血糖")
                recommendations.append("定期监测胎儿发育情况")

            # 年龄相关建议
            if age > 65:
                recommendations.append("老年糖尿病患者需注意低血糖风险")
                recommendations.append("适当放宽血糖控制目标")

        except Exception as e:
            logger.warning(f"Failed to get clinical guidelines: {e}")

        return recommendations

    def _get_evidence_based_treatments(self, patient_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """获取循证医学支持的治疗建议"""
        treatments = []

        try:
            diabetes_type = patient_info.get('diabetes_type', 'type2')
            bmi = patient_info.get('bmi', 25)

            # 基于BMI和糖尿病类型的治疗建议
            if diabetes_type == 'type2' and bmi > 30:
                treatments.append({
                    'name': 'GLP-1受体激动剂',
                    'evidence_level': 'A',
                    'rationale': '肥胖型2型糖尿病的一线选择，具有减重和心血管保护作用'
                })
                treatments.append({
                    'name': 'SGLT2抑制剂',
                    'evidence_level': 'A',
                    'rationale': '具有心血管和肾脏保护作用，适合合并心血管疾病的患者'
                })
            elif diabetes_type == 'type1':
                treatments.append({
                    'name': '基础-餐时胰岛素方案',
                    'evidence_level': 'A',
                    'rationale': '1型糖尿病的标准治疗方案，提供最佳的血糖控制'
                })

        except Exception as e:
            logger.warning(f"Failed to get evidence-based treatments: {e}")

        return treatments

    def _get_complication_prevention(self, patient_info: Dict[str, Any]) -> List[str]:
        """获取并发症预防建议"""
        prevention_measures = []

        try:
            diabetes_duration = patient_info.get('duration_years', 5)
            age = patient_info.get('age', 50)

            # 基于病程的并发症预防
            if diabetes_duration > 10:
                prevention_measures.append("长期糖尿病患者需定期筛查视网膜病变")
                prevention_measures.append("建议每年进行肾功能检查")
                prevention_measures.append("注意足部护理，预防糖尿病足")

            # 基于年龄的预防措施
            if age > 60:
                prevention_measures.append("老年患者需特别注意心血管疾病预防")
                prevention_measures.append("定期监测血压和血脂")

            # 通用预防措施
            prevention_measures.append("维持良好的血糖控制(HbA1c<7%)")
            prevention_measures.append("定期进行并发症筛查")

        except Exception as e:
            logger.warning(f"Failed to get complication prevention measures: {e}")

        return prevention_measures

class DiabReKGAdapter(BaseKGAdapter):
    """DiabRe-KG适配器"""

    def __init__(self, config: KGConfig):
        super().__init__(config)
        self.kg_data = None
        if config.diabre_kg_path:
            self._load_kg_data()

    def _load_kg_data(self):
        """加载DiabRe-KG数据"""
        try:
            if self.config.diabre_kg_path.endswith('.csv'):
                self.kg_data = pd.read_csv(self.config.diabre_kg_path)
            elif self.config.diabre_kg_path.endswith('.json'):
                with open(self.config.diabre_kg_path, 'r', encoding='utf-8') as f:
                    self.kg_data = json.load(f)
            logger.info("Successfully loaded DiabRe-KG data")
        except Exception as e:
            logger.warning(f"Failed to load DiabRe-KG data: {e}")

    def get_patient_features(self, patient_id: str, patient_info: Dict[str, Any]) -> PatientKGFeatures:
        """获取患者DiabRe-KG特征"""
        features = PatientKGFeatures()

        if not self.kg_data:
            return features

        try:
            # 获取足溃疡史等风险因子
            features.complication_risk_factors.extend(
                self._get_foot_ulcer_risk_factors(patient_info)
            )

        except Exception as e:
            logger.error(f"Error retrieving DiabRe-KG features for patient {patient_id}: {e}")

        return features

    def _get_foot_ulcer_risk_factors(self, patient_info: Dict[str, Any]) -> List[str]:
        """获取足溃疡风险因子"""
        risk_factors = []

        try:
            if isinstance(self.kg_data, pd.DataFrame):
                # 基于患者特征查询风险因子
                age = patient_info.get('age', 50)
                diabetes_duration = patient_info.get('duration_years', 5)

                # 查询相关风险因子
                risk_data = self.kg_data[
                    (self.kg_data['age_group'].str.contains(str(age // 10 * 10))) &
                    (self.kg_data['diabetes_duration'] >= diabetes_duration)
                ]

                for _, row in risk_data.iterrows():
                    if row['risk_factor'] not in risk_factors:
                        risk_factors.append(row['risk_factor'])

        except Exception as e:
            logger.warning(f"Failed to get foot ulcer risk factors: {e}")

        return risk_factors

class OpenFDAAdapter(BaseKGAdapter):
    """OpenFDA适配器"""

    def __init__(self, config: KGConfig):
        super().__init__(config)
        self.session = requests.Session()
        if config.openfda_api_key:
            self.session.headers.update({'Authorization': f'Bearer {config.openfda_api_key}'})

    def get_patient_features(self, patient_id: str, patient_info: Dict[str, Any]) -> PatientKGFeatures:
        """获取患者OpenFDA特征"""
        features = PatientKGFeatures()

        try:
            # 获取药物不良反应数据
            features.drug_interactions.extend(
                self._get_drug_adverse_events(patient_info)
            )

            # 获取药物标签信息
            features.dietary_recommendations.extend(
                self._get_drug_labeling_info(patient_info)
            )

        except Exception as e:
            logger.error(f"Error retrieving OpenFDA features for patient {patient_id}: {e}")

        return features

    def _get_drug_adverse_events(self, patient_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """获取药物不良反应数据"""
        adverse_events = []

        try:
            # 模拟OpenFDA API调用获取糖尿病药物不良反应
            diabetes_drugs = ['metformin', 'insulin', 'glipizide', 'glyburide']

            for drug in diabetes_drugs:
                # 模拟API响应
                adverse_event = {
                    'drug_name': drug,
                    'adverse_events': [
                        {
                            'event': 'hypoglycemia',
                            'frequency': 'common',
                            'severity': 'moderate'
                        },
                        {
                            'event': 'gastrointestinal_upset',
                            'frequency': 'common',
                            'severity': 'mild'
                        }
                    ],
                    'source': 'OpenFDA'
                }
                adverse_events.append(adverse_event)

        except Exception as e:
            logger.warning(f"Failed to get drug adverse events: {e}")

        return adverse_events

    def _get_drug_labeling_info(self, patient_info: Dict[str, Any]) -> List[str]:
        """获取药物标签信息"""
        labeling_info = []

        try:
            diabetes_type = patient_info.get('diabetes_type', 'type2')

            if diabetes_type == 'type1':
                labeling_info.append("胰岛素是1型糖尿病的必需治疗药物")
                labeling_info.append("注意监测血糖，避免低血糖事件")
            elif diabetes_type == 'type2':
                labeling_info.append("二甲双胍是2型糖尿病的一线治疗药物")
                labeling_info.append("注意肾功能监测，避免乳酸酸中毒")
            elif diabetes_type == 'gdm':
                labeling_info.append("妊娠期糖尿病需严格控制血糖")
                labeling_info.append("优先考虑胰岛素治疗，避免口服降糖药")

        except Exception as e:
            logger.warning(f"Failed to get drug labeling info: {e}")

        return labeling_info

class ClinicalTrialsAdapter(BaseKGAdapter):
    """ClinicalTrials.gov适配器"""

    def __init__(self, config: KGConfig):
        super().__init__(config)
        self.session = requests.Session()

    def get_patient_features(self, patient_id: str, patient_info: Dict[str, Any]) -> PatientKGFeatures:
        """获取患者临床试验特征"""
        features = PatientKGFeatures()

        try:
            # 获取相关临床试验信息
            features.dietary_recommendations.extend(
                self._get_clinical_trial_info(patient_info)
            )

            # 获取试验结果和疗效数据
            features.complication_risk_factors.extend(
                self._get_trial_outcomes(patient_info)
            )

        except Exception as e:
            logger.error(f"Error retrieving ClinicalTrials features for patient {patient_id}: {e}")

        return features

    def _get_clinical_trial_info(self, patient_info: Dict[str, Any]) -> List[str]:
        """获取临床试验信息"""
        trial_info = []

        try:
            diabetes_type = patient_info.get('diabetes_type', 'type2')
            age = patient_info.get('age', 50)

            # 基于患者特征推荐相关临床试验
            if diabetes_type == 'type1' and age < 30:
                trial_info.append("1型糖尿病青少年患者可考虑参与CGM优化试验")
                trial_info.append("新型胰岛素类似物临床试验可能适合")
            elif diabetes_type == 'type2' and age > 60:
                trial_info.append("老年2型糖尿病患者可考虑参与心血管保护试验")
                trial_info.append("SGLT2抑制剂心血管结局试验")
            elif diabetes_type == 'gdm':
                trial_info.append("妊娠期糖尿病患者可考虑参与血糖控制试验")
                trial_info.append("产后糖尿病预防试验")

        except Exception as e:
            logger.warning(f"Failed to get clinical trial info: {e}")

        return trial_info

    def _get_trial_outcomes(self, patient_info: Dict[str, Any]) -> List[str]:
        """获取试验结果和疗效数据"""
        outcomes = []

        try:
            diabetes_type = patient_info.get('diabetes_type', 'type2')

            if diabetes_type == 'type1':
                outcomes.append("CGM使用可显著降低HbA1c 0.5-1.0%")
                outcomes.append("闭环胰岛素泵系统改善血糖控制")
            elif diabetes_type == 'type2':
                outcomes.append("SGLT2抑制剂降低心血管事件风险30%")
                outcomes.append("GLP-1受体激动剂减重效果显著")
            elif diabetes_type == 'gdm':
                outcomes.append("严格血糖控制降低巨大儿风险")
                outcomes.append("产后生活方式干预预防2型糖尿病")

        except Exception as e:
            logger.warning(f"Failed to get trial outcomes: {e}")

        return outcomes

class USDANutritionAdapter(BaseKGAdapter):
    """USDA营养数据库适配器"""

    def __init__(self, config: KGConfig):
        super().__init__(config)
        self.session = requests.Session()
        if config.usda_api_key:
            self.session.headers.update({'X-API-Key': config.usda_api_key})

    def get_patient_features(self, patient_id: str, patient_info: Dict[str, Any]) -> PatientKGFeatures:
        """获取患者USDA营养特征"""
        features = PatientKGFeatures()

        try:
            # 获取食物营养成分数据
            features.dietary_recommendations.extend(
                self._get_food_nutrition_info(patient_info)
            )

            # 获取血糖指数信息
            features.complication_risk_factors.extend(
                self._get_glycemic_index_info(patient_info)
            )

        except Exception as e:
            logger.error(f"Error retrieving USDA features for patient {patient_id}: {e}")

        return features

    def _get_food_nutrition_info(self, patient_info: Dict[str, Any]) -> List[str]:
        """获取食物营养成分信息"""
        nutrition_info = []

        try:
            diabetes_type = patient_info.get('diabetes_type', 'type2')

            if diabetes_type == 'type1':
                nutrition_info.append("1型糖尿病患者需注意碳水化合物计数")
                nutrition_info.append("建议选择低GI食物，如燕麦、全麦面包")
                nutrition_info.append("注意蛋白质和脂肪的平衡摄入")
            elif diabetes_type == 'type2':
                nutrition_info.append("2型糖尿病患者需控制总热量摄入")
                nutrition_info.append("增加膳食纤维摄入，如蔬菜、水果")
                nutrition_info.append("限制精制碳水化合物，选择复合碳水化合物")
            elif diabetes_type == 'gdm':
                nutrition_info.append("妊娠期糖尿病患者需均衡营养")
                nutrition_info.append("增加叶酸、铁、钙等营养素摄入")
                nutrition_info.append("控制餐后血糖，选择低GI食物")

        except Exception as e:
            logger.warning(f"Failed to get food nutrition info: {e}")

        return nutrition_info

    def _get_glycemic_index_info(self, patient_info: Dict[str, Any]) -> List[str]:
        """获取血糖指数信息"""
        gi_info = []

        try:
            # 基于USDA数据库的GI信息
            gi_info.append("低GI食物(GI<55): 燕麦、豆类、大部分蔬菜")
            gi_info.append("中等GI食物(GI 55-70): 全麦面包、糙米、香蕉")
            gi_info.append("高GI食物(GI>70): 白面包、白米、糖果")
            gi_info.append("建议糖尿病患者选择低GI食物，有助于血糖控制")

        except Exception as e:
            logger.warning(f"Failed to get glycemic index info: {e}")

        return gi_info

class WHOGHOAdapter(BaseKGAdapter):
    """WHO全球健康观测站适配器"""

    def __init__(self, config: KGConfig):
        super().__init__(config)
        self.session = requests.Session()

    def get_patient_features(self, patient_id: str, patient_info: Dict[str, Any]) -> PatientKGFeatures:
        """获取患者WHO健康数据特征"""
        features = PatientKGFeatures()

        try:
            # 获取全球糖尿病流行病学数据
            features.complication_risk_factors.extend(
                self._get_diabetes_epidemiology(patient_info)
            )

            # 获取健康政策建议
            features.dietary_recommendations.extend(
                self._get_health_policy_recommendations(patient_info)
            )

        except Exception as e:
            logger.error(f"Error retrieving WHO features for patient {patient_id}: {e}")

        return features

    def _get_diabetes_epidemiology(self, patient_info: Dict[str, Any]) -> List[str]:
        """获取糖尿病流行病学数据"""
        epidemiology_info = []

        try:
            age = patient_info.get('age', 50)
            diabetes_type = patient_info.get('diabetes_type', 'type2')

            if diabetes_type == 'type1':
                epidemiology_info.append("1型糖尿病全球患病率约0.1-0.3%")
                epidemiology_info.append("1型糖尿病多见于儿童和青少年")
                epidemiology_info.append("1型糖尿病与自身免疫因素相关")
            elif diabetes_type == 'type2':
                epidemiology_info.append("2型糖尿病全球患病率约8-10%")
                epidemiology_info.append("2型糖尿病多见于中老年人")
                epidemiology_info.append("2型糖尿病与生活方式因素密切相关")
            elif diabetes_type == 'gdm':
                epidemiology_info.append("妊娠期糖尿病患病率约2-10%")
                epidemiology_info.append("GDM增加产后2型糖尿病风险")
                epidemiology_info.append("GDM需要严格的血糖控制")

            # 年龄相关风险
            if age > 65:
                epidemiology_info.append("老年糖尿病患者并发症风险增加")
                epidemiology_info.append("老年患者需注意低血糖风险")

        except Exception as e:
            logger.warning(f"Failed to get diabetes epidemiology: {e}")

        return epidemiology_info

    def _get_health_policy_recommendations(self, patient_info: Dict[str, Any]) -> List[str]:
        """获取健康政策建议"""
        policy_recommendations = []

        try:
            diabetes_type = patient_info.get('diabetes_type', 'type2')

            if diabetes_type == 'type1':
                policy_recommendations.append("WHO建议1型糖尿病患者使用胰岛素治疗")
                policy_recommendations.append("建议定期监测血糖和HbA1c")
                policy_recommendations.append("推荐使用连续血糖监测(CGM)")
            elif diabetes_type == 'type2':
                policy_recommendations.append("WHO建议2型糖尿病患者生活方式干预")
                policy_recommendations.append("推荐二甲双胍作为一线治疗药物")
                policy_recommendations.append("建议定期筛查并发症")
            elif diabetes_type == 'gdm':
                policy_recommendations.append("WHO建议GDM患者严格控制血糖")
                policy_recommendations.append("推荐产后糖尿病筛查")
                policy_recommendations.append("建议生活方式干预预防2型糖尿病")

        except Exception as e:
            logger.warning(f"Failed to get health policy recommendations: {e}")

        return policy_recommendations

class DataSourceQualityEvaluator:
    """数据源质量评估器"""

    def __init__(self):
        self.quality_threshold = 0.6  # 质量阈值
        self.logger = logging.getLogger(__name__)

    def evaluate_data_source(self, source_name: str, features: PatientKGFeatures) -> DataSourceQuality:
        """评估数据源质量"""
        # 基于特征内容评估质量
        relevance_score = self._calculate_relevance_score(source_name, features)
        completeness_score = self._calculate_completeness_score(features)
        accuracy_score = self._calculate_accuracy_score(source_name, features)
        timeliness_score = self._calculate_timeliness_score(source_name)

        return DataSourceQuality(
            source_name=source_name,
            relevance_score=relevance_score,
            completeness_score=completeness_score,
            accuracy_score=accuracy_score,
            timeliness_score=timeliness_score,
            overall_quality=0.0  # 将在__post_init__中计算
        )

    def _calculate_relevance_score(self, source_name: str, features: PatientKGFeatures) -> float:
        """计算相关性评分"""
        relevance_scores = {
            'ohdsi': 0.9,  # 高度相关
            'kg_cn': 0.8,  # 高度相关
            'ruijin': 0.85,  # 高度相关
            'diabre': 0.7,  # 中等相关
            'openfda': 0.75,  # 中等相关
            'clinicaltrials': 0.8,  # 高度相关
            'usda': 0.6,  # 中等相关
            'who': 0.5   # 低相关
        }
        return relevance_scores.get(source_name, 0.5)

    def _calculate_completeness_score(self, features: PatientKGFeatures) -> float:
        """计算完整性评分"""
        total_features = 0
        available_features = 0

        if features.icr_ratio is not None:
            total_features += 1
            available_features += 1

        if features.regional_gi_preferences:
            total_features += 1
            available_features += 1

        if features.complication_risk_factors:
            total_features += 1
            available_features += 1

        if features.drug_interactions:
            total_features += 1
            available_features += 1

        if features.dietary_recommendations:
            total_features += 1
            available_features += 1

        if features.exercise_recommendations:
            total_features += 1
            available_features += 1

        return available_features / max(total_features, 1)

    def _calculate_accuracy_score(self, source_name: str, features: PatientKGFeatures) -> float:
        """计算准确性评分"""
        accuracy_scores = {
            'ohdsi': 0.9,  # 高准确性
            'kg_cn': 0.8,  # 高准确性
            'ruijin': 0.85,  # 高准确性
            'diabre': 0.7,  # 中等准确性
            'openfda': 0.8,  # 高准确性
            'clinicaltrials': 0.75,  # 中等准确性
            'usda': 0.9,  # 高准确性
            'who': 0.8   # 高准确性
        }
        return accuracy_scores.get(source_name, 0.5)

    def _calculate_timeliness_score(self, source_name: str) -> float:
        """计算时效性评分"""
        timeliness_scores = {
            'ohdsi': 0.8,  # 高时效性
            'kg_cn': 0.7,  # 中等时效性
            'ruijin': 0.6,  # 中等时效性
            'diabre': 0.5,  # 低时效性
            'openfda': 0.9,  # 高时效性
            'clinicaltrials': 0.8,  # 高时效性
            'usda': 0.8,  # 高时效性
            'who': 0.7   # 中等时效性
        }
        return timeliness_scores.get(source_name, 0.5)

    def filter_high_quality_sources(self, quality_scores: Dict[str, DataSourceQuality]) -> List[str]:
        """过滤高质量数据源"""
        high_quality_sources = []
        for source_name, quality in quality_scores.items():
            if quality.overall_quality >= self.quality_threshold:
                high_quality_sources.append(source_name)
                self.logger.info(f"✅ {source_name}: 质量评分 {quality.overall_quality:.3f} (通过)")
            else:
                self.logger.warning(f"❌ {source_name}: 质量评分 {quality.overall_quality:.3f} (未通过)")

        return high_quality_sources

class IntegratedKGAdapter:
    """集成知识图谱适配器"""

    def __init__(self, config: KGConfig):
        self.config = config
        self.adapters = {
            'ohdsi': OHSDIAdapter(config),
            'kg_cn': DiabetesKGCNAdapter(config),
            'ruijin': RuijinKGAdapter(config),
            'diabre': DiabReKGAdapter(config),
            'openfda': OpenFDAAdapter(config),
            'clinicaltrials': ClinicalTrialsAdapter(config),
            'usda': USDANutritionAdapter(config),
            'who': WHOGHOAdapter(config)
        }
        self.quality_evaluator = DataSourceQualityEvaluator()
        self.logger = logging.getLogger(__name__)

    def get_patient_features(self, patient_id: str, patient_info: Dict[str, Any]) -> PatientKGFeatures:
        """获取患者综合知识图谱特征（质量过滤版本）"""
        integrated_features = PatientKGFeatures()

        # 从各个适配器获取特征并评估质量
        for adapter_name, adapter in self.adapters.items():
            try:
                features = adapter.get_patient_features(patient_id, patient_info)

                # 评估数据源质量
                quality = self.quality_evaluator.evaluate_data_source(adapter_name, features)

                # 只使用高质量数据源的特征
                if quality.overall_quality >= self.quality_evaluator.quality_threshold:
                    self._merge_features(integrated_features, features)
                    # 记录特征质量评分
                    integrated_features.feature_quality_scores[adapter_name] = quality.overall_quality
                    self.logger.info(f"✅ 使用 {adapter_name} 数据源 (质量: {quality.overall_quality:.3f})")
                else:
                    self.logger.warning(f"❌ 跳过 {adapter_name} 数据源 (质量: {quality.overall_quality:.3f})")

            except Exception as e:
                self.logger.warning(f"Failed to get features from {adapter_name}: {e}")

        # 记录质量评估结果
        self.logger.info(f"📊 数据源质量评估完成，共使用 {len(integrated_features.feature_quality_scores)} 个高质量数据源")

        return integrated_features

    def get_quality_filtered_features(self, patient_id: str, patient_info: Dict[str, Any]) -> Tuple[PatientKGFeatures, Dict[str, DataSourceQuality]]:
        """获取质量过滤后的特征和质量评估结果"""
        integrated_features = PatientKGFeatures()
        quality_scores = {}

        # 从各个适配器获取特征并评估质量
        for adapter_name, adapter in self.adapters.items():
            try:
                features = adapter.get_patient_features(patient_id, patient_info)

                # 评估数据源质量
                quality = self.quality_evaluator.evaluate_data_source(adapter_name, features)
                quality_scores[adapter_name] = quality

                # 只使用高质量数据源的特征
                if quality.overall_quality >= self.quality_evaluator.quality_threshold:
                    self._merge_features(integrated_features, features)
                    # 记录特征质量评分
                    integrated_features.feature_quality_scores[adapter_name] = quality.overall_quality

            except Exception as e:
                self.logger.warning(f"Failed to get features from {adapter_name}: {e}")

        return integrated_features, quality_scores

    def _merge_features(self, target: PatientKGFeatures, source: PatientKGFeatures):
        """合并特征"""
        # 合并ICR比值
        if source.icr_ratio is not None:
            target.icr_ratio = source.icr_ratio

        # 合并地域GI偏好
        target.regional_gi_preferences.update(source.regional_gi_preferences)

        # 合并并发症风险因子
        target.complication_risk_factors.extend(source.complication_risk_factors)

        # 合并药物相互作用
        target.drug_interactions.extend(source.drug_interactions)

        # 合并饮食建议
        target.dietary_recommendations.extend(source.dietary_recommendations)

        # 合并运动建议
        target.exercise_recommendations.extend(source.exercise_recommendations)

    def get_enhanced_features_vector(self, patient_id: str, patient_info: Dict[str, Any]) -> np.ndarray:
        """获取增强特征向量"""
        features = self.get_patient_features(patient_id, patient_info)

        # 构建特征向量
        feature_vector = []

        # ICR比值特征
        if features.icr_ratio is not None:
            feature_vector.append(features.icr_ratio)
        else:
            feature_vector.append(0.0)  # 默认值

        # 地域GI偏好特征（取平均值）
        if features.regional_gi_preferences:
            avg_gi = np.mean(list(features.regional_gi_preferences.values()))
            feature_vector.append(avg_gi)
        else:
            feature_vector.append(50.0)  # 默认GI值

        # 并发症风险因子数量
        feature_vector.append(len(features.complication_risk_factors))

        # 药物相互作用数量
        feature_vector.append(len(features.drug_interactions))

        # 饮食建议数量
        feature_vector.append(len(features.dietary_recommendations))

        # 运动建议数量
        feature_vector.append(len(features.exercise_recommendations))

        return np.array(feature_vector, dtype=np.float32)

    def get_clinical_insights(self, patient_id: str, patient_info: Dict[str, Any]) -> Dict[str, Any]:
        """获取临床洞察"""
        features = self.get_patient_features(patient_id, patient_info)

        insights = {
            'icr_analysis': self._analyze_icr(features.icr_ratio),
            'dietary_analysis': self._analyze_dietary_preferences(features.regional_gi_preferences),
            'risk_assessment': self._assess_complication_risks(features.complication_risk_factors),
            'drug_safety': self._analyze_drug_interactions(features.drug_interactions),
            'lifestyle_recommendations': {
                'dietary': features.dietary_recommendations,
                'exercise': features.exercise_recommendations
            }
        }

        return insights

    def _analyze_icr(self, icr_ratio: Optional[float]) -> Dict[str, Any]:
        """分析ICR比值"""
        if icr_ratio is None:
            return {'status': 'unknown', 'recommendation': '需要进一步评估ICR比值'}

        if icr_ratio < 5:
            return {'status': 'low', 'recommendation': 'ICR比值较低，可能需要调整胰岛素剂量'}
        elif icr_ratio > 15:
            return {'status': 'high', 'recommendation': 'ICR比值较高，可能需要增加胰岛素剂量'}
        else:
            return {'status': 'normal', 'recommendation': 'ICR比值在正常范围内'}

    def _analyze_dietary_preferences(self, gi_preferences: Dict[str, float]) -> Dict[str, Any]:
        """分析饮食偏好"""
        if not gi_preferences:
            return {'status': 'unknown', 'recommendation': '需要获取地域饮食偏好信息'}

        avg_gi = np.mean(list(gi_preferences.values()))

        if avg_gi > 70:
            return {'status': 'high_gi', 'recommendation': '地域饮食偏好高GI食物，建议选择低GI替代品'}
        elif avg_gi < 55:
            return {'status': 'low_gi', 'recommendation': '地域饮食偏好低GI食物，有利于血糖控制'}
        else:
            return {'status': 'moderate_gi', 'recommendation': '地域饮食偏好中等GI食物，需要平衡选择'}

    def _assess_complication_risks(self, risk_factors: List[str]) -> Dict[str, Any]:
        """评估并发症风险"""
        if not risk_factors:
            return {'status': 'low', 'recommendation': '未发现明显并发症风险因子'}

        high_risk_factors = [factor for factor in risk_factors if 'severe' in factor.lower() or 'high' in factor.lower()]

        if len(high_risk_factors) > 2:
            return {'status': 'high', 'recommendation': '存在多个高风险因子，需要密切监测'}
        elif len(high_risk_factors) > 0:
            return {'status': 'moderate', 'recommendation': '存在中等风险因子，需要定期评估'}
        else:
            return {'status': 'low', 'recommendation': '风险因子较少，继续当前管理策略'}

    def _analyze_drug_interactions(self, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析药物相互作用"""
        if not interactions:
            return {'status': 'safe', 'recommendation': '未发现明显药物相互作用'}

        return {'status': 'monitor', 'recommendation': f'发现{len(interactions)}种药物，需要监测相互作用'}

# 工厂函数
def create_kg_adapter(config: KGConfig) -> IntegratedKGAdapter:
    """创建知识图谱适配器"""
    return IntegratedKGAdapter(config)

# 默认配置
DEFAULT_KG_CONFIG = KGConfig(
    ohdsi_base_url="https://fhir.ohdsi.org/api",
    kg_cn_url="http://kg-diabetes.cn:7474",
    kg_cn_username="neo4j",
    kg_cn_password="password",
    ruijin_kg_path="data/ruijin_diabetes_kg.json",
    diabre_kg_path="data/diabre_kg.csv",
    openfda_base_url="https://api.fda.gov",
    clinicaltrials_base_url="https://clinicaltrials.gov/api/v2",
    usda_base_url="https://api.nal.usda.gov/fdc/v1",
    who_base_url="https://apps.who.int/gho/athena/api",
    cache_enabled=True,
    cache_ttl_hours=24
)

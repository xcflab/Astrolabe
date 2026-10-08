#!/usr/bin/env python3
"""
配体氢键饱和度分析和VDM采样过滤器

功能：
1. 检测配体中每个化学基团的氢键饱和状态
2. 为未饱和基团筛选合适距离范围内的残基用于VDM采样

基于 test_buried_unsat_hbonds 的方法论：
- 距离范围：4.5-7.0 Å（基于文献）
  - 下限 4.5 Å：太近无法采样不同构象
  - 上限 7.0 Å：超过则无法形成有效氢键
- 使用CB原子作为参考点（GLY用CA）
- 排除已与配体形成氢键的残基
"""

import sys
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
import pyrosetta
from pyrosetta import rosetta


class LigandHBondFilter:
    """配体氢键分析和VDM采样过滤器"""
    
    def __init__(self, 
                 pose,
                 ligand_chain: str = "C",
                 distance_min: float = 4.5,
                 distance_max: float = 7.0,
                 use_cb: bool = True,
                 exclude_hbond_residues: bool = True):
        """
        初始化过滤器
        
        Args:
            pose: PyRosetta Pose对象
            ligand_chain: 配体链ID
            distance_min: 最小距离阈值（Å）
            distance_max: 最大距离阈值（Å）
            use_cb: 是否使用CB原子计算距离（否则用最近重原子）
            exclude_hbond_residues: 是否排除已形成氢键的残基
        """
        self.pose = pose
        self.ligand_chain = ligand_chain
        self.distance_min = distance_min
        self.distance_max = distance_max
        self.use_cb = use_cb
        self.exclude_hbond_residues = exclude_hbond_residues
        
        # 识别配体残基
        self.ligand_residues = self._identify_ligand_residues()
        
        # 分析氢键
        self.hbond_residues = set()
        self.hbond_satisfied_atoms = set()  # 已饱和的配体原子
        self._analyze_ligand_hbonds()
    
    def _identify_ligand_residues(self) -> List[int]:
        """识别配体残基索引"""
        ligand_res = []
        pdb_info = self.pose.pdb_info()
        
        if not pdb_info:
            print("警告: PDB信息不可用")
            return ligand_res
        
        for res_idx in range(1, self.pose.total_residue() + 1):
            chain = pdb_info.chain(res_idx)
            if chain == self.ligand_chain:
                ligand_res.append(res_idx)
        
        print(f"识别到 {len(ligand_res)} 个配体残基（链 {self.ligand_chain}）")
        return ligand_res
    
    def _analyze_ligand_hbonds(self):
        """分析配体的氢键，识别已饱和的原子"""
        if not self.ligand_residues:
            print("警告: 未找到配体残基")
            return
        
        # 计算氢键
        hbond_set = rosetta.core.scoring.hbonds.HBondSet()
        self.pose.update_residue_neighbors()
        rosetta.core.scoring.hbonds.fill_hbond_set(
            self.pose, False, hbond_set
        )
        
        hbond_residues = set()
        satisfied_atoms = set()  # 格式: (res_idx, atom_name)
        
        # 遍历所有氢键
        for i in range(1, hbond_set.nhbonds() + 1):
            hbond = hbond_set.hbond(i)
            don_res = hbond.don_res()
            acc_res = hbond.acc_res()
            
            # 配体是供体
            if don_res in self.ligand_residues:
                don_hatm = hbond.don_hatm()
                residue = self.pose.residue(don_res)
                atom_name = residue.atom_name(don_hatm).strip()
                satisfied_atoms.add((don_res, atom_name))
                
                # 记录蛋白质受体残基
                if acc_res not in self.ligand_residues:
                    protein_res = self.pose.residue(acc_res)
                    if protein_res.is_protein():
                        hbond_residues.add(acc_res)
            
            # 配体是受体
            elif acc_res in self.ligand_residues:
                acc_atm = hbond.acc_atm()
                residue = self.pose.residue(acc_res)
                atom_name = residue.atom_name(acc_atm).strip()
                satisfied_atoms.add((acc_res, atom_name))
                
                # 记录蛋白质供体残基
                if don_res not in self.ligand_residues:
                    protein_res = self.pose.residue(don_res)
                    if protein_res.is_protein():
                        hbond_residues.add(don_res)
        
        self.hbond_residues = hbond_residues
        self.hbond_satisfied_atoms = satisfied_atoms
        
        print(f"配体形成 {len(satisfied_atoms)} 个氢键（原子级）")
        print(f"涉及 {len(hbond_residues)} 个蛋白质残基")
    
    def is_group_saturated(self, ligand_atom_names: List[str]) -> bool:
        """
        检查化学基团是否已饱和
        
        Args:
            ligand_atom_names: 基团中的原子名称列表
            
        Returns:
            True if 基团中任何关键原子已形成氢键
        """
        # 检查基团中是否有原子已形成氢键
        for res_idx in self.ligand_residues:
            for atom_name in ligand_atom_names:
                if (res_idx, atom_name) in self.hbond_satisfied_atoms:
                    return True
        return False
    
    def get_eligible_residues_for_group(
        self, 
        represent_atom_name: str,
        ligand_res_idx: Optional[int] = None
    ) -> List[Dict]:
        """
        获取适合对特定基团进行VDM采样的残基
        
        Args:
            represent_atom_name: 基团的代表性原子名称（通常是供体/受体原子）
            ligand_res_idx: 配体残基索引（如果为None，使用第一个配体残基）
            
        Returns:
            符合条件的残基列表，每个包含：
                - residue_num: 残基索引
                - residue_name: 残基名称
                - chain: 链ID
                - pdb_num: PDB编号
                - distance: 到代表性原子的距离
        """
        if not ligand_res_idx:
            if not self.ligand_residues:
                return []
            ligand_res_idx = self.ligand_residues[0]
        
        # 获取代表性原子坐标
        ligand_res = self.pose.residue(ligand_res_idx)
        
        if not ligand_res.has(represent_atom_name):
            print(f"警告: 配体残基 {ligand_res_idx} 没有原子 {represent_atom_name}")
            return []
        
        ref_xyz = ligand_res.xyz(represent_atom_name)
        ref_coord = (ref_xyz.x, ref_xyz.y, ref_xyz.z)
        
        eligible_residues = []
        pdb_info = self.pose.pdb_info()
        
        # 遍历所有蛋白质残基
        for res_idx in range(1, self.pose.total_residue() + 1):
            residue = self.pose.residue(res_idx)
            
            # 跳过配体和非蛋白质残基
            if res_idx in self.ligand_residues or not residue.is_protein():
                continue
            
            # 排除已与配体形成氢键的残基
            if self.exclude_hbond_residues and res_idx in self.hbond_residues:
                continue
            
            # 计算距离
            if self.use_cb:
                # 使用CB（GLY用CA）
                ref_atom = "CA" if residue.name3() == "GLY" else "CB"
                
                if residue.has(ref_atom):
                    xyz = residue.xyz(ref_atom)
                    coord = (xyz.x, xyz.y, xyz.z)
                    distance = self._calculate_distance(ref_coord, coord)
                    
                    # 检查距离范围
                    if self.distance_min <= distance <= self.distance_max:
                        eligible_residues.append({
                            'residue_num': res_idx,
                            'residue_name': residue.name3(),
                            'chain': pdb_info.chain(res_idx) if pdb_info else 'X',
                            'pdb_num': pdb_info.number(res_idx) if pdb_info else res_idx,
                            'distance': distance,
                            'reference_atom': ref_atom
                        })
            else:
                # 使用最近重原子
                min_distance = float('inf')
                closest_atom = None
                
                for atom_idx in range(1, residue.natoms() + 1):
                    if residue.atom_type(atom_idx).element() == "H":
                        continue
                    
                    xyz = residue.xyz(atom_idx)
                    coord = (xyz.x, xyz.y, xyz.z)
                    distance = self._calculate_distance(ref_coord, coord)
                    
                    if distance < min_distance:
                        min_distance = distance
                        closest_atom = residue.atom_name(atom_idx).strip()
                
                # 检查距离范围
                if self.distance_min <= min_distance <= self.distance_max:
                    eligible_residues.append({
                        'residue_num': res_idx,
                        'residue_name': residue.name3(),
                        'chain': pdb_info.chain(res_idx) if pdb_info else 'X',
                        'pdb_num': pdb_info.number(res_idx) if pdb_info else res_idx,
                        'distance': min_distance,
                        'reference_atom': closest_atom
                    })
        
        return eligible_residues
    
    @staticmethod
    def _calculate_distance(coord1: Tuple, coord2: Tuple) -> float:
        """计算两点距离"""
        return ((coord1[0] - coord2[0])**2 + 
                (coord1[1] - coord2[1])**2 + 
                (coord1[2] - coord2[2])**2)**0.5


def filter_vdm_sampling_targets(
    pose,
    vdm_cg_aa_atommap_dict: Dict,
    ligand_chain: str = "C",
    distance_min: float = 4.5,
    distance_max: float = 7.0,
    verbose: bool = True
) -> Tuple[Dict, Dict]:
    """
    过滤VDM采样目标，排除已饱和基团并限制残基范围
    
    Args:
        pose: PyRosetta Pose对象
        vdm_cg_aa_atommap_dict: VDM化学基团映射字典
        ligand_chain: 配体链ID
        distance_min: 最小距离阈值
        distance_max: 最大距离阈值
        verbose: 是否打印详细信息
        
    Returns:
        (filtered_cg_dict, eligible_residues_dict)
        - filtered_cg_dict: 过滤后的CG映射（排除已饱和基团）
        - eligible_residues_dict: 每个CG对应的合格残基 {cg_key: [res_info]}
    """
    # 初始化过滤器
    hbond_filter = LigandHBondFilter(
        pose=pose,
        ligand_chain=ligand_chain,
        distance_min=distance_min,
        distance_max=distance_max
    )
    
    filtered_cg_dict = {}
    eligible_residues_dict = {}
    
    saturated_count = 0
    unsaturated_count = 0
    
    if verbose:
        print("\n" + "=" * 80)
        print("VDM采样目标过滤")
        print("=" * 80)
    
    # 遍历所有CG
    for cg_key, cg_info in vdm_cg_aa_atommap_dict.items():
        lgd_atoms = cg_info['lgd_sel']
        represent_atom = cg_info['represent_name']
        cg_name = cg_info['cg']
        
        # 检查是否饱和
        is_saturated = hbond_filter.is_group_saturated(lgd_atoms)
        
        if is_saturated:
            saturated_count += 1
            if verbose:
                print(f"跳过 CG {cg_key} ({cg_name}): 基团已饱和 {lgd_atoms}")
            continue
        
        query_atom = represent_atom
        ligand_res_idx = None
        if lgd_atoms:
            satisfied_atoms = hbond_filter.hbond_satisfied_atoms
            ligand_res_indices = hbond_filter.ligand_residues
            for res_idx in ligand_res_indices:
                for atom_name in lgd_atoms:
                    if (res_idx, atom_name) not in satisfied_atoms:
                        query_atom = atom_name
                        ligand_res_idx = res_idx
                        break
                if ligand_res_idx is not None:
                    break
            else:
                query_atom = lgd_atoms[-1]
        
        eligible_res = hbond_filter.get_eligible_residues_for_group(
            query_atom,
            ligand_res_idx=ligand_res_idx
        )
        
        if not eligible_res:
            if verbose:
                print(f"跳过 CG {cg_key} ({cg_name}): 无合格残基（距离 {distance_min}-{distance_max} Å）")
            continue
        
        # 保留此CG
        filtered_cg_dict[cg_key] = cg_info
        eligible_residues_dict[cg_key] = eligible_res
        unsaturated_count += 1
        
        if verbose:
            print(f"保留 CG {cg_key} ({cg_name}): {len(eligible_res)} 个合格残基")
            if len(eligible_res) <= 5:
                for res in eligible_res:
                    print(f"  - {res['residue_name']}{res['pdb_num']}:{res['chain']} "
                          f"(距离={res['distance']:.2f}Å)")
    
    if verbose:
        print("\n" + "-" * 80)
        print(f"过滤结果:")
        print(f"  已饱和基团: {saturated_count}")
        print(f"  未饱和基团: {unsaturated_count}")
        print(f"  保留用于VDM采样: {len(filtered_cg_dict)}")
        print("=" * 80 + "\n")
    
    return filtered_cg_dict, eligible_residues_dict

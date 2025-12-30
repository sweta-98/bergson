from typing import List, Tuple
from abc import ABC

debug = False


class AlignmentStrategy(ABC):
    def align_tokens(
        self,
        source_tokens: list[int],
        target_tokens: list[int],
    ) -> list[int]:
        """
        Abstract method to generate a mapping from Source Token Index -> Target Token Index.
        
        Args:
            source_tokens: The reference sequence.
            target_tokens: The sequence to align against.
        
        """
        raise NotImplementedError("align_tokens method must be implemented by subclasses.")


class SnapAlignmentStrategy(AlignmentStrategy):
    """
    Snap alignment strategy.
    
    Generates a mapping from Source indices to Target indices.
    """
    
    def __init__(self, min_anchor_length: int = 1):
        self.min_anchor_length = min_anchor_length
    
    def _find_anchors(
        self,
        source_tokens: List[int],
        target_tokens: List[int],
    ) -> List[Tuple[int, int, int]]:
        if debug:
            print(f"  [DEBUG] Finding anchors (Min len: {self.min_anchor_length})...")
        n, m = len(source_tokens), len(target_tokens)
        
        # DP Table construction
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        matches = []
        
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if source_tokens[i - 1] == target_tokens[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                    if dp[i][j] >= self.min_anchor_length:
                        matches.append((i, j, dp[i][j]))
        
        # Sort by length descending
        matches.sort(key=lambda x: -x[2])
        
        anchors = []
        source_used = [False] * n
        target_used = [False] * m
        
        for source_end, target_end, length in matches:
            source_start = source_end - length
            target_start = target_end - length
            
            source_overlap = any(source_used[i] for i in range(source_start, source_end))
            target_overlap = any(target_used[i] for i in range(target_start, target_end))
            
            if not source_overlap and not target_overlap:
                anchors.append((source_start, target_start, length))
                for i in range(source_start, source_end):
                    source_used[i] = True
                for i in range(target_start, target_end):
                    target_used[i] = True
        
        # Sort by source position
        anchors.sort(key=lambda x: x[0])
        
        if debug:
            print(f"  [DEBUG] Found {len(anchors)} anchors.")
            for idx, a in enumerate(anchors):
                print(f"    -> Anchor {idx}: Source[{a[0]}:{a[0]+a[2]}] maps to Target[{a[1]}:{a[1]+a[2]}]")
            
        return anchors
    
    def _snap_align_stretch(
        self,
        source_len: int,
        target_len: int,
        fallback_index: int
    ) -> List[int]:
        """
        Calculates relative target indices for a gap.
        Returns a list of integer offsets relative to the start of the target gap.
        """
        if source_len == 0:
            return []
            
        # If target gap is empty (deletion), snap all source tokens to the fallback index
        if target_len == 0:
            if debug:
                print(f"    [DEBUG] Empty Target Gap. Mapping {source_len} source tokens to index {fallback_index}.")
            return [fallback_index] * source_len
        

        if debug:
            print(f"    [DEBUG] Snapping Gap: Squashing {source_len} source tokens into {target_len} target tokens.")
        alignment = []
        for i in range(source_len):
            frac = i / source_len if source_len > 1 else 0.5
            target_pos = int(frac * target_len)
            target_pos = min(target_pos, target_len - 1)
            alignment.append(target_pos)
        return alignment
    
    def align_tokens(
        self,
        source_tokens: list[int],
        target_tokens: list[int],
    ) -> list[int]:
        """
        Generate a mapping from Source Token Index -> Target Token Index.
        
        Args:
            source_tokens: The reference sequence.
            target_tokens: The sequence to align against.
            
        Returns:
            List[int]: A list of length len(source_tokens).
                       result[i] is the index j in target_tokens that aligns with source_tokens[i].
                       
        Raises:
            ValueError: If either sequence is empty.
        """
        # 1. Validation
        if not source_tokens or not target_tokens:
            raise ValueError("Alignment failed: Source and Target sequences must not be empty.")

        if debug:
            print(f"\n[DEBUG] Starting Alignment. Source Len: {len(source_tokens)}, Target Len: {len(target_tokens)}")
        
        anchors = self._find_anchors(source_tokens, target_tokens)
        
        # Result array: maps source index -> target index
        alignment_map = [-1] * len(source_tokens)
        
        prev_source_end = 0
        prev_target_end = 0
        
        # Iterate through anchors and fill gaps
        for i, (source_start, target_start, length) in enumerate(anchors):
            # --- GAP HANDLING (Before Anchor) ---
            if prev_source_end < source_start:
                gap_source_len = source_start - prev_source_end
                gap_target_len = target_start - prev_target_end
                
                # If target gap is empty, fallback to the last valid target token
                fallback_idx = max(0, prev_target_end - 1)
                
                if debug:
                    print(f"  [DEBUG] Processing GAP before Anchor {i}")
                
                if gap_target_len == 0:
                    # Direct mapping to fallback
                    for k in range(gap_source_len):
                        alignment_map[prev_source_end + k] = fallback_idx
                else:
                    # Interpolation
                    gap_offsets = self._snap_align_stretch(
                        gap_source_len, 
                        gap_target_len, 
                        fallback_index=fallback_idx # Unused in this branch
                    )
                    for k, rel_offset in enumerate(gap_offsets):
                        alignment_map[prev_source_end + k] = prev_target_end + rel_offset
            
            # --- ANCHOR HANDLING ---
            if debug:
                print(f"  [DEBUG] Locking in Anchor {i}")
            for k in range(length):
                alignment_map[source_start + k] = target_start + k
            
            prev_source_end = source_start + length
            prev_target_end = target_start + length
        
        # --- TRAILING GAP HANDLING ---
        if prev_source_end < len(source_tokens):
            if debug:
                print(f"  [DEBUG] Processing trailing GAP")
            gap_source_len = len(source_tokens) - prev_source_end
            gap_target_len = len(target_tokens) - prev_target_end
            
            fallback_idx = max(0, prev_target_end - 1)
            
            if gap_target_len == 0:
                for k in range(gap_source_len):
                    alignment_map[prev_source_end + k] = fallback_idx
            else:
                gap_offsets = self._snap_align_stretch(
                    gap_source_len, 
                    gap_target_len, 
                    fallback_index=fallback_idx
                )
                for k, rel_offset in enumerate(gap_offsets):
                    alignment_map[prev_source_end + k] = prev_target_end + rel_offset
        
        if debug:
            print(f"[DEBUG] Finished. Map Length: {len(alignment_map)}")

        return alignment_map
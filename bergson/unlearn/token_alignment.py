"""
Token alignment strategies for matching tokens between aligned datasets.

This module provides strategies for aligning tokens between datasets where
examples are aligned by index (same DOI, same title) but content may differ.
"""

from typing import List, Tuple, Optional, Protocol


# class TokenAlignmentStrategy(Protocol):
#     """Protocol for token alignment strategies."""
    
#     def align_tokens(
#         self,
#         source_tokens: List[int],
#         target_tokens: List[int],
#         source_attention_mask: Optional[List[int]] = None,
#         target_attention_mask: Optional[List[int]] = None,
#     ) -> Tuple[List[int], List[int], List[int], List[int]]:
#         """
#         Align tokens between source and target sequences.
        
#         Args:
#             source_tokens: Token IDs from source sequence
#             target_tokens: Token IDs from target sequence
#             source_attention_mask: Optional attention mask for source
#             target_attention_mask: Optional attention mask for target
            
#         Returns:
#             Tuple of (aligned_source_tokens, aligned_target_tokens,
#                      aligned_source_mask, aligned_target_mask)
#         """
#         ...


# class PositionalAlignmentStrategy:
#     """
#     Simple positional alignment strategy.
    
#     Aligns tokens by position, truncating to the minimum length.
#     This assumes similar structure between source and target.
#     """
    
#     def align_tokens(
#         self,
#         source_tokens: List[int],
#         target_tokens: List[int],
#         source_attention_mask: Optional[List[int]] = None,
#         target_attention_mask: Optional[List[int]] = None,
#     ) -> Tuple[List[int], List[int], List[int], List[int]]:
#         """
#         Align tokens by position, using minimum length.
#         """
#         min_len = min(len(source_tokens), len(target_tokens))
        
#         aligned_source = source_tokens[:min_len]
#         aligned_target = target_tokens[:min_len]
        
#         if source_attention_mask is None:
#             aligned_source_mask = [1] * min_len
#         else:
#             aligned_source_mask = source_attention_mask[:min_len]
            
#         if target_attention_mask is None:
#             aligned_target_mask = [1] * min_len
#         else:
#             aligned_target_mask = target_attention_mask[:min_len]
        
#         return aligned_source, aligned_target, aligned_source_mask, aligned_target_mask


# class PrefixAlignmentStrategy:
#     """
#     Prefix alignment strategy.
    
#     Aligns tokens by matching a common prefix (e.g., title/header).
#     Falls back to positional alignment if prefix matching fails.
#     """
    
#     def __init__(self, prefix_length: int = 50):
#         """
#         Args:
#             prefix_length: Length of prefix to match
#         """
#         self.prefix_length = prefix_length
    
#     def align_tokens(
#         self,
#         source_tokens: List[int],
#         target_tokens: List[int],
#         source_attention_mask: Optional[List[int]] = None,
#         target_attention_mask: Optional[List[int]] = None,
#     ) -> Tuple[List[int], List[int], List[int], List[int]]:
#         """
#         Align tokens by matching prefix, then positional alignment.
#         """
#         # Try to find matching prefix
#         prefix_len = min(self.prefix_length, len(source_tokens), len(target_tokens))
#         source_prefix = source_tokens[:prefix_len]
#         target_prefix = target_tokens[:prefix_len]
        
#         # If prefixes match, use positional alignment from there
#         if source_prefix == target_prefix:
#             # Use full sequences with positional alignment
#             min_len = min(len(source_tokens), len(target_tokens))
#             aligned_source = source_tokens[:min_len]
#             aligned_target = target_tokens[:min_len]
#         else:
#             # Fallback to simple positional alignment
#             min_len = min(len(source_tokens), len(target_tokens))
#             aligned_source = source_tokens[:min_len]
#             aligned_target = target_tokens[:min_len]
        
#         if source_attention_mask is None:
#             aligned_source_mask = [1] * len(aligned_source)
#         else:
#             aligned_source_mask = source_attention_mask[:len(aligned_source)]
            
#         if target_attention_mask is None:
#             aligned_target_mask = [1] * len(aligned_target)
#         else:
#             aligned_target_mask = target_attention_mask[:len(aligned_target)]
        
#         return aligned_source, aligned_target, aligned_source_mask, aligned_target_mask


# def align_token_sequences(
#     source_tokens: List[int],
#     target_tokens: List[int],
#     strategy: TokenAlignmentStrategy,
#     source_attention_mask: Optional[List[int]] = None,
#     target_attention_mask: Optional[List[int]] = None,
# ) -> Tuple[List[int], List[int], List[int], List[int]]:
#     """
#     Convenience function to align token sequences using a strategy.
    
#     Args:
#         source_tokens: Token IDs from source sequence
#         target_tokens: Token IDs from target sequence
#         strategy: Alignment strategy to use
#         source_attention_mask: Optional attention mask for source
#         target_attention_mask: Optional attention mask for target
        
#     Returns:
#         Tuple of (aligned_source_tokens, aligned_target_tokens,
#                  aligned_source_mask, aligned_target_mask)
#     """
#     return strategy.align_tokens(
#         source_tokens,
#         target_tokens,
#         source_attention_mask,
#         target_attention_mask,
#     )



class SnapAlignmentStrategy:
    """
    Snap alignment strategy.
    
    Finds identical "anchor" stretches between sequences and aligns them 1:1.
    For differing stretches between anchors, aligns tokens by their fractional
    position within the stretch (snapping to the nearest target token).
    """
    
    def __init__(self, min_anchor_length: int = 1):
        """
        Args:
            min_anchor_length: Minimum length of matching tokens to consider an anchor
        """
        self.min_anchor_length = min_anchor_length
    
    def _find_anchors(
        self,
        source_tokens: List[int],
        target_tokens: List[int],
    ) -> List[Tuple[int, int, int]]:
        """
        Find matching anchor regions using longest common substring approach.
        
        Returns:
            List of (source_start, target_start, length) tuples for matching regions,
            sorted by source position.
        """
        # Use dynamic programming to find all common substrings
        # Then greedily select non-overlapping anchors
        
        n, m = len(source_tokens), len(target_tokens)
        if n == 0 or m == 0:
            return []
        
        # Build table of match lengths ending at each position
        # dp[i][j] = length of common substring ending at source[i-1], target[j-1]
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        matches = []  # (source_end, target_end, length)
        
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if source_tokens[i - 1] == target_tokens[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                    if dp[i][j] >= self.min_anchor_length:
                        matches.append((i, j, dp[i][j]))
        
        # Filter to get maximal matches (not subsumed by longer ones)
        # and select non-overlapping anchors greedily by length
        matches.sort(key=lambda x: -x[2])  # Sort by length descending
        
        anchors = []
        source_used = [False] * n
        target_used = [False] * m
        
        for source_end, target_end, length in matches:
            source_start = source_end - length
            target_start = target_end - length
            
            # Check if this region overlaps with already selected anchors
            source_overlap = any(source_used[i] for i in range(source_start, source_end))
            target_overlap = any(target_used[i] for i in range(target_start, target_end))
            
            if not source_overlap and not target_overlap:
                anchors.append((source_start, target_start, length))
                for i in range(source_start, source_end):
                    source_used[i] = True
                for i in range(target_start, target_end):
                    target_used[i] = True
        
        # Sort anchors by source position
        anchors.sort(key=lambda x: x[0])
        return anchors
    
    def _snap_align_stretch(
        self,
        source_len: int,
        target_len: int,
    ) -> List[int]:
        """
        For a differing stretch, compute target index for each source position
        based on fractional position.
        
        Args:
            source_len: Length of source stretch
            target_len: Length of target stretch
            
        Returns:
            List of target indices (one per source position)
        """
        if source_len == 0:
            return []
        if target_len == 0:
            # No target tokens to align to - return -1 as sentinel
            return [-1] * source_len
        
        alignment = []
        for i in range(source_len):
            # Fractional position in source (0 to 1)
            frac = i / source_len if source_len > 1 else 0.5
            # Map to target position and snap to nearest
            target_pos = int(frac * target_len)
            target_pos = min(target_pos, target_len - 1)  # Clamp to valid range
            alignment.append(target_pos)
        
        return alignment
    
    def align_tokens(
        self,
        source_tokens: List[int],
        target_tokens: List[int],
        source_attention_mask: Optional[List[int]] = None,
        target_attention_mask: Optional[List[int]] = None,
    ) -> Tuple[List[int], List[int], List[int], List[int]]:
        """
        Align tokens using anchor-based snap alignment.
        
        Returns aligned sequences where each source token is paired with
        a corresponding target token (may have duplicates in target).
        """
        anchors = self._find_anchors(source_tokens, target_tokens)
        
        # Build alignment: for each source position, what target position?
        source_to_target = [-1] * len(source_tokens)
        
        # Process regions between anchors
        prev_source_end = 0
        prev_target_end = 0
        
        for source_start, target_start, length in anchors:
            # Handle the gap before this anchor
            if prev_source_end < source_start:
                gap_source_len = source_start - prev_source_end
                gap_target_len = target_start - prev_target_end
                
                gap_alignment = self._snap_align_stretch(gap_source_len, gap_target_len)
                
                for i, rel_target in enumerate(gap_alignment):
                    if rel_target >= 0:
                        source_to_target[prev_source_end + i] = prev_target_end + rel_target
            
            # Handle the anchor (1:1 alignment)
            for i in range(length):
                source_to_target[source_start + i] = target_start + i
            
            prev_source_end = source_start + length
            prev_target_end = target_start + length
        
        # Handle any trailing gap after last anchor
        if prev_source_end < len(source_tokens):
            gap_source_len = len(source_tokens) - prev_source_end
            gap_target_len = len(target_tokens) - prev_target_end
            
            gap_alignment = self._snap_align_stretch(gap_source_len, gap_target_len)
            
            for i, rel_target in enumerate(gap_alignment):
                if rel_target >= 0:
                    source_to_target[prev_source_end + i] = prev_target_end + rel_target
        
        # Build output sequences
        aligned_source = []
        aligned_target = []
        aligned_source_mask = []
        aligned_target_mask = []
        
        for src_idx, tgt_idx in enumerate(source_to_target):
            if tgt_idx >= 0:
                aligned_source.append(source_tokens[src_idx])
                aligned_target.append(target_tokens[tgt_idx])
                
                if source_attention_mask is not None:
                    aligned_source_mask.append(source_attention_mask[src_idx])
                else:
                    aligned_source_mask.append(1)
                
                if target_attention_mask is not None:
                    aligned_target_mask.append(target_attention_mask[tgt_idx])
                else:
                    aligned_target_mask.append(1)
        
        return aligned_source, aligned_target, aligned_source_mask, aligned_target_mask
    
    def get_alignment_map(
        self,
        source_tokens: List[int],
        target_tokens: List[int],
    ) -> List[int]:
        """
        Get the raw alignment map from source indices to target indices.
        
        Useful for debugging or when you need the mapping itself.
        
        Returns:
            List where result[i] is the target index aligned to source index i,
            or -1 if no alignment exists.
        """
        anchors = self._find_anchors(source_tokens, target_tokens)
        source_to_target = [-1] * len(source_tokens)
        
        prev_source_end = 0
        prev_target_end = 0
        
        for source_start, target_start, length in anchors:
            if prev_source_end < source_start:
                gap_source_len = source_start - prev_source_end
                gap_target_len = target_start - prev_target_end
                gap_alignment = self._snap_align_stretch(gap_source_len, gap_target_len)
                for i, rel_target in enumerate(gap_alignment):
                    if rel_target >= 0:
                        source_to_target[prev_source_end + i] = prev_target_end + rel_target
            
            for i in range(length):
                source_to_target[source_start + i] = target_start + i
            
            prev_source_end = source_start + length
            prev_target_end = target_start + length
        
        if prev_source_end < len(source_tokens):
            gap_source_len = len(source_tokens) - prev_source_end
            gap_target_len = len(target_tokens) - prev_target_end
            gap_alignment = self._snap_align_stretch(gap_source_len, gap_target_len)
            for i, rel_target in enumerate(gap_alignment):
                if rel_target >= 0:
                    source_to_target[prev_source_end + i] = prev_target_end + rel_target
        
        return source_to_target
#!/usr/bin/env python3
"""
Label positions with Stockfish at depth 7.
Produces best move and evaluation for each position.
"""

import argparse
import json
import chess
import chess.engine
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing


PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9
}

PIECE_NAMES = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king"
}


def describe_move(board: chess.Board, move_uci: str) -> str:
    """
    Describe a move in plain English.
    E.g., "White pawn from e2 moves to e4" or "Black knight from g8 captures pawn on f6"
    """
    try:
        move = chess.Move.from_uci(move_uci)
        
        from_sq = chess.square_name(move.from_square)
        to_sq = chess.square_name(move.to_square)
        
        piece = board.piece_at(move.from_square)
        if piece is None:
            return f"moves {move_uci}"
        
        color_name = "White" if piece.color == chess.WHITE else "Black"
        piece_name = PIECE_NAMES.get(piece.piece_type, "piece")
        
        # Check for castling
        if piece.piece_type == chess.KING:
            if move_uci in ["e1g1", "e8g8"]:
                return f"{color_name} castles kingside"
            elif move_uci in ["e1c1", "e8c8"]:
                return f"{color_name} castles queenside"
        
        # Check for capture
        captured = board.piece_at(move.to_square)
        is_en_passant = board.is_en_passant(move)
        
        if captured or is_en_passant:
            if is_en_passant:
                captured_name = "pawn"
            else:
                captured_name = PIECE_NAMES.get(captured.piece_type, "piece")
            
            if piece.piece_type == chess.PAWN:
                desc = f"{color_name} pawn from {from_sq} captures {captured_name} on {to_sq}"
            else:
                desc = f"{color_name} {piece_name} from {from_sq} captures {captured_name} on {to_sq}"
        else:
            # Regular move
            if piece.piece_type == chess.PAWN:
                rank_diff = abs(int(to_sq[1]) - int(from_sq[1]))
                if rank_diff == 2:
                    desc = f"{color_name} pawn from {from_sq} moves two squares to {to_sq}"
                else:
                    desc = f"{color_name} pawn from {from_sq} moves to {to_sq}"
            else:
                desc = f"{color_name} {piece_name} from {from_sq} moves to {to_sq}"
        
        # Check for promotion
        if move.promotion:
            promoted_name = PIECE_NAMES.get(move.promotion, "queen")
            desc = f"{color_name} pawn from {from_sq} promotes to {promoted_name} on {to_sq}"
            if captured:
                captured_name = PIECE_NAMES.get(captured.piece_type, "piece")
                desc = f"{color_name} pawn from {from_sq} captures {captured_name} and promotes to {promoted_name} on {to_sq}"
        
        # Add check info
        board_copy = board.copy()
        board_copy.push(move)
        if board_copy.is_checkmate():
            desc += " - Checkmate!"
        elif board_copy.is_check():
            desc += " - Check!"
        
        return desc
        
    except Exception as e:
        return f"plays {move_uci}"


def count_material(board: chess.Board) -> int:
    """Count material advantage for side to move."""
    stm = board.turn
    stm_material = sum(len(board.pieces(pt, stm)) * val for pt, val in PIECE_VALUES.items())
    opp_material = sum(len(board.pieces(pt, not stm)) * val for pt, val in PIECE_VALUES.items())
    return stm_material - opp_material


def get_game_phase(board: chess.Board) -> str:
    """Determine game phase."""
    total_pieces = len(board.piece_map())
    queens = len(board.pieces(chess.QUEEN, chess.WHITE)) + len(board.pieces(chess.QUEEN, chess.BLACK))
    
    if board.fullmove_number <= 10 and total_pieces >= 28:
        return "opening"
    elif total_pieces >= 16 or queens >= 2:
        return "middlegame"
    else:
        return "endgame"


def analyze_position(
    board: chess.Board, 
    engine: chess.engine.SimpleEngine, 
    depth: int = 7
) -> Optional[Dict]:
    """
    Analyze position with Stockfish at specified depth.
    Returns best move and evaluation.
    """
    try:
        # Get analysis with top move
        info = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=1)
        
        if not info:
            return None
        
        # Handle case where info is a list
        entry = info[0] if isinstance(info, list) else info
        
        if "pv" not in entry or not entry["pv"]:
            return None
        
        best_move = entry["pv"][0]
        score = entry["score"].relative
        
        # Convert score to centipawns
        if score.is_mate():
            mate_in = score.mate()
            eval_cp = 10000 if mate_in > 0 else -10000
            eval_str = f"M{abs(mate_in)}" if mate_in > 0 else f"-M{abs(mate_in)}"
        else:
            eval_cp = score.score()
            eval_str = f"{eval_cp:+d}"
        
        return {
            "best_move": best_move.uci(),
            "eval_cp": eval_cp,
            "eval_str": eval_str,
            "depth": depth
        }
    
    except Exception as e:
        return None


def create_training_example(
    position_data: Dict,
    analysis: Dict,
    include_eval_in_prompt: bool = False
) -> Dict:
    """
    Create a training example in chat format.
    
    Args:
        position_data: Original position data with FEN
        analysis: Stockfish analysis results
        include_eval_in_prompt: Whether to include evaluation in the response
    """
    fen = position_data["fen"]
    board = chess.Board(fen)
    
    legal_moves = [m.uci() for m in board.legal_moves]
    legal_moves_str = " ".join(legal_moves)
    
    best_move = analysis["best_move"]
    eval_str = analysis["eval_str"]
    eval_cp = analysis["eval_cp"]
    
    # Build prompt
    prompt = f"""You are an expert chess player. Here is the position in FEN format:
{fen}

Legal moves: {legal_moves_str}

Select the best move. Keep your thinking to 2 sentences or less, then output your chosen move.
Format:
<think>brief thinking (2 sentences max)</think>
<uci_move>your_move</uci_move>"""

    # Build response with move description
    move_description = describe_move(board, best_move)
    response = f"<think>{move_description}.</think><uci_move>{best_move}</uci_move>"
    
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]
    }


def label_positions(
    input_file: str,
    output_file: str,
    stockfish_path: str,
    depth: int = 7,
    batch_size: int = 1000
):
    """
    Label all positions in input file with Stockfish analysis.
    """
    print(f"Labeling positions from {input_file}")
    print(f"Stockfish depth: {depth}")
    print(f"Output: {output_file}")
    
    # Count input lines
    with open(input_file, 'r') as f:
        total_lines = sum(1 for _ in f)
    print(f"Total positions to label: {total_lines}")
    
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    
    # Process positions
    labeled_count = 0
    skipped_count = 0
    
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        pbar = tqdm(total=total_lines, desc="Labeling positions")
        
        for line in fin:
            pbar.update(1)
            
            try:
                position_data = json.loads(line.strip())
                fen = position_data["fen"]
                board = chess.Board(fen)
                
                # Skip invalid positions
                if board.is_game_over() or len(list(board.legal_moves)) < 2:
                    skipped_count += 1
                    continue
                
                # Analyze with Stockfish
                analysis = analyze_position(board, engine, depth=depth)
                
                if analysis is None:
                    skipped_count += 1
                    continue
                
                # Verify best move is legal
                if analysis["best_move"] not in [m.uci() for m in board.legal_moves]:
                    skipped_count += 1
                    continue
                
                # Create training example
                example = create_training_example(position_data, analysis)
                
                # Add metadata
                example["metadata"] = {
                    "source": position_data.get("source", "unknown"),
                    "phase": position_data.get("phase", get_game_phase(board)),
                    "eval_cp": analysis["eval_cp"],
                    "best_move": analysis["best_move"],
                    "depth": depth
                }
                
                fout.write(json.dumps(example) + '\n')
                labeled_count += 1
                
                if labeled_count % 10000 == 0:
                    pbar.set_postfix({"labeled": labeled_count, "skipped": skipped_count})
                
            except Exception as e:
                skipped_count += 1
                continue
        
        pbar.close()
    
    engine.quit()
    
    print(f"\nLabeling complete!")
    print(f"Labeled: {labeled_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Output saved to: {output_file}")
    
    return labeled_count


def label_puzzles(
    input_file: str,
    output_file: str,
    stockfish_path: str,
    depth: int = 7
):
    """
    Label puzzle positions - uses the first position FEN and solution's first move.
    For puzzles, we might want to use the puzzle's solution move rather than SF's best move,
    but we'll verify with SF that it's a good move.
    """
    print(f"Labeling puzzle positions from {input_file}")
    print(f"Stockfish depth: {depth}")
    
    with open(input_file, 'r') as f:
        total_lines = sum(1 for _ in f)
    print(f"Total puzzles to label: {total_lines}")
    
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    
    labeled_count = 0
    skipped_count = 0
    
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        pbar = tqdm(total=total_lines, desc="Labeling puzzles")
        
        for line in fin:
            pbar.update(1)
            
            try:
                puzzle_data = json.loads(line.strip())
                fen = puzzle_data["fen"]
                board = chess.Board(fen)
                
                # For puzzles, the position is BEFORE the opponent's move
                # The puzzle starts after opponent moves, so we need to apply first move
                moves = puzzle_data.get("moves", "").split()
                
                if not moves:
                    skipped_count += 1
                    continue
                
                # Apply opponent's move to get to the puzzle position
                try:
                    opp_move = chess.Move.from_uci(moves[0])
                    board.push(opp_move)
                except:
                    skipped_count += 1
                    continue
                
                if board.is_game_over() or len(list(board.legal_moves)) < 2:
                    skipped_count += 1
                    continue
                
                # Now analyze the puzzle position
                analysis = analyze_position(board, engine, depth=depth)
                
                if analysis is None:
                    skipped_count += 1
                    continue
                
                # The puzzle solution move (what the player should find)
                if len(moves) > 1:
                    puzzle_solution = moves[1]
                    # Verify it's legal
                    if puzzle_solution not in [m.uci() for m in board.legal_moves]:
                        # Fall back to SF best move
                        puzzle_solution = analysis["best_move"]
                else:
                    puzzle_solution = analysis["best_move"]
                
                # Create position data for training example
                position_data = {
                    "fen": board.fen(),
                    "source": puzzle_data.get("source", "lichess_puzzle"),
                    "phase": "puzzle",
                    "rating": puzzle_data.get("rating", 0),
                    "themes": puzzle_data.get("themes", "")
                }
                
                # Use the puzzle solution or SF best move
                analysis_for_training = analysis.copy()
                analysis_for_training["best_move"] = puzzle_solution
                
                example = create_training_example(position_data, analysis_for_training)
                
                example["metadata"] = {
                    "source": "lichess_puzzle",
                    "phase": "puzzle",
                    "puzzle_type": puzzle_data.get("puzzle_type", "unknown"),
                    "rating": puzzle_data.get("rating", 0),
                    "themes": puzzle_data.get("themes", ""),
                    "eval_cp": analysis["eval_cp"],
                    "best_move": puzzle_solution,
                    "sf_best_move": analysis["best_move"],
                    "depth": depth
                }
                
                fout.write(json.dumps(example) + '\n')
                labeled_count += 1
                
            except Exception as e:
                skipped_count += 1
                continue
        
        pbar.close()
    
    engine.quit()
    
    print(f"\nLabeling complete!")
    print(f"Labeled: {labeled_count}")
    print(f"Skipped: {skipped_count}")
    
    return labeled_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label positions with Stockfish analysis")
    parser.add_argument("--input", type=str, required=True,
                        help="Input JSONL file with positions")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSONL file with labeled data")
    parser.add_argument("--stockfish", type=str, default="/opt/homebrew/bin/stockfish",
                        help="Path to Stockfish binary")
    parser.add_argument("--depth", type=int, default=7,
                        help="Stockfish analysis depth")
    parser.add_argument("--puzzles", action="store_true",
                        help="Input is puzzle data (needs special handling)")
    args = parser.parse_args()
    
    if args.puzzles:
        label_puzzles(
            input_file=args.input,
            output_file=args.output,
            stockfish_path=args.stockfish,
            depth=args.depth
        )
    else:
        label_positions(
            input_file=args.input,
            output_file=args.output,
            stockfish_path=args.stockfish,
            depth=args.depth
        )

import pygame
import chess
import math


WIDTH, HEIGHT = 1500, 940

BG = (24, 25, 32)
PANEL = (35, 37, 47)
TEXT = (238, 239, 245)
MUTED = (150, 154, 169)

# Cores exatas do tabuleiro da imagem (Lilás / Lavanda e Cinza-Gelo)
BOARD_DARK = (134, 121, 189)
BOARD_LIGHT = (240, 240, 242)

ACCENT = (105, 190, 130)
GOLD = (238, 190, 85)
RED = (220, 90, 90)
BLUE = (90, 160, 230)

# Glifos preenchidos (preto sólido) usados como máscara para desenhar contorno + preenchimento limpo
SOLID_GLYPH = {
    chess.KING: "♚",
    chess.QUEEN: "♛",
    chess.ROOK: "♜",
    chess.BISHOP: "♝",
    chess.KNIGHT: "♞",
    chess.PAWN: "♟",
}

OUTLINE_GLYPH = {
    chess.KING: "♔",
    chess.QUEEN: "♕",
    chess.ROOK: "♖",
    chess.BISHOP: "♗",
    chess.KNIGHT: "♘",
    chess.PAWN: "♙",
}


class ChessUI:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Neural Chess — Genetic Self Play")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("segoe ui", 20)
        self.small = pygame.font.SysFont("segoe ui", 14)
        self.tiny = pygame.font.SysFont("segoe ui", 11)
        self.coord_font_cache = {}
        self._piece_sprites = {}
        self.count = 25
        self.buttons = []
        self.mini_rects = []
        self.focus_id = 1
        self.scroll = 0

    def text(self, s, x, y, color=TEXT, font=None):
        font = font or self.font
        self.screen.blit(font.render(str(s), True, color), (int(x), int(y)))

    def _get_coord_font(self, sq):
        px = max(9, int(sq * 0.21))
        if px not in self.coord_font_cache:
            self.coord_font_cache[px] = pygame.font.SysFont("segoe ui", px, bold=True)
        return self.coord_font_cache[px]

    def _get_piece_sprite(self, piece_type, is_white, sq):
        """Gera sprite nítido em cache com preenchimento branco + contorno preto forte (estilo vetor)."""
        key = (piece_type, is_white, int(sq))
        if key in self._piece_sprites:
            return self._piece_sprites[key]

        size = max(16, int(sq))
        font_px = max(14, int(sq * 0.84))
        font = pygame.font.SysFont("segoe ui symbol", font_px)

        surf = pygame.Surface((size, size), pygame.SRCALPHA)
        solid = font.render(SOLID_GLYPH[piece_type], True, (255, 255, 255) if is_white else (20, 20, 24))
        rect = solid.get_rect(center=(size // 2, size // 2))

        if is_white:
            # Contorno preto ao redor da silhueta branca para ficar idêntico ao estilo vetorial da foto
            outline_mask = font.render(SOLID_GLYPH[piece_type], True, (18, 18, 22))
            for ox, oy in ((-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1), (0, 2)):
                surf.blit(outline_mask, (rect.x + ox, rect.y + oy))
            surf.blit(solid, rect)
            # Detalhes internos pretos (cruz do rei/bispo, coroa da dama, base)
            inner_lines = font.render(OUTLINE_GLYPH[piece_type], True, (22, 22, 26))
            surf.blit(inner_lines, rect)
        else:
            # Peça preta sólida com leve brilho interno branco para destacar os detalhes
            surf.blit(solid, rect)
            inner_highlight = font.render(OUTLINE_GLYPH[piece_type], True, (235, 235, 240))
            inner_highlight.set_alpha(150)
            surf.blit(inner_highlight, rect)

        self._piece_sprites[key] = surf
        return surf

    def board(self, b, x, y, size, mini=False, last_move=None, best_move=None):
        sq = size / 8
        coord_font = self._get_coord_font(sq) if not mini else None

        for r in range(8):
            for c in range(8):
                square = chess.square(c, 7 - r)
                is_light = (r + c) % 2 == 0
                col = BOARD_LIGHT if is_light else BOARD_DARK

                rect = pygame.Rect(
                    int(x + c * sq),
                    int(y + r * sq),
                    math.ceil(sq),
                    math.ceil(sq),
                )
                pygame.draw.rect(self.screen, col, rect)

                if last_move and square in (last_move.from_square, last_move.to_square):
                    overlay = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
                    overlay.fill((255, 225, 90, 110))
                    self.screen.blit(overlay, rect.topleft)

                # Coordenadas 1..8 na coluna 'a' e a..h na linha '1' (idêntico à foto)
                if not mini:
                    label_col = BOARD_DARK if is_light else BOARD_LIGHT
                    if c == 0:
                        rank_lbl = coord_font.render(str(8 - r), True, label_col)
                        self.screen.blit(rank_lbl, (rect.x + 3, rect.y + 2))
                    if r == 7:
                        file_lbl = coord_font.render(chr(ord("a") + c), True, label_col)
                        self.screen.blit(
                            file_lbl,
                            (rect.right - file_lbl.get_width() - 3, rect.bottom - file_lbl.get_height() - 1),
                        )

                if isinstance(b, dict):
                    piece_data = b.get(square)
                    if piece_data is None:
                        piece_type = piece_color = None
                    else:
                        piece_type, piece_color = piece_data
                else:
                    piece = b.piece_at(square)
                    piece_type = piece.piece_type if piece else None
                    piece_color = piece.color if piece else None

                if piece_type is not None:
                    if mini and size < 76:
                        color = (252, 252, 254) if piece_color else (24, 24, 28)
                        radius = max(2, int(sq * 0.30))
                        center = (int(rect.centerx), int(rect.centery))
                        pygame.draw.circle(self.screen, color, center, radius)
                        pygame.draw.circle(self.screen, (30, 30, 35), center, radius, 1)
                    else:
                        sprite = self._get_piece_sprite(piece_type, bool(piece_color), sq)
                        self.screen.blit(sprite, sprite.get_rect(center=rect.center))

        if not mini:
            # Borda fina ao redor do tabuleiro principal
            pygame.draw.rect(self.screen, (60, 58, 74), (int(x), int(y), int(size), int(size)), 1)
            if last_move:
                self.draw_arrow(last_move, x, y, size, (245, 205, 65, 175), 6)
            if best_move:
                self.draw_arrow(best_move, x, y, size, (70, 160, 245, 155), 4)

    def draw_arrow(self, move, x, y, size, color, thickness=5):
        if move is None:
            return
        sq = size / 8

        def center(square):
            file = chess.square_file(square)
            rank = 7 - chess.square_rank(square)
            return (int(x + (file + 0.5) * sq), int(y + (rank + 0.5) * sq))

        start = center(move.from_square)
        end = center(move.to_square)
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)
        if length <= 0:
            return

        ux = dx / length
        uy = dy / length
        head_length = max(10, size * 0.055)
        end_line = (int(end[0] - ux * head_length), int(end[1] - uy * head_length))

        overlay = pygame.Surface((self.screen.get_width(), self.screen.get_height()), pygame.SRCALPHA)
        pygame.draw.line(overlay, color, start, end_line, thickness)
        perpendicular = (-uy, ux)
        p1 = (int(end[0]), int(end[1]))
        p2 = (
            int(end[0] - ux * head_length + perpendicular[0] * head_length * 0.55),
            int(end[1] - uy * head_length + perpendicular[1] * head_length * 0.55),
        )
        p3 = (
            int(end[0] - ux * head_length - perpendicular[0] * head_length * 0.55),
            int(end[1] - uy * head_length - perpendicular[1] * head_length * 0.55),
        )
        pygame.draw.polygon(overlay, color, [p1, p2, p3])
        self.screen.blit(overlay, (0, 0))

    def draw_graph(self, history, x, y, w, h):
        pygame.draw.rect(self.screen, PANEL, (x, y, w, h), border_radius=10)
        self.text("EVOLUÇÃO DO RATING", x + 12, y + 7, TEXT, self.small)
        if len(history) < 2:
            self.text("Aguardando dados de evolução...", x + 12, y + 35, MUTED, self.small)
            return

        points = history[-100:]
        values = [float(item[1]) for item in points]
        minimum = min(values)
        maximum = max(values)
        if maximum == minimum:
            maximum += 1

        left = x + 14
        right = x + w - 14
        top = y + 29
        bottom = y + h - 17
        graph_h = max(1, bottom - top)
        graph_w = max(1, right - left)

        coords = []
        for i, value in enumerate(values):
            px = left + (i / max(1, len(values) - 1)) * graph_w
            py = bottom - ((value - minimum) / (maximum - minimum)) * graph_h
            coords.append((int(px), int(py)))

        if len(coords) >= 2:
            pygame.draw.lines(self.screen, ACCENT, False, coords, 2)

        self.text(f"{values[-1]:.0f}", right - 30, top - 17, GOLD, self.small)
        self.text(f"{minimum:.0f}", left, bottom + 2, MUTED, self.tiny)

    def draw_result_stats(self, totals, recent_results, x, y, w, h=76):
        panel = pygame.Rect(x, y, w, h)
        pygame.draw.rect(self.screen, PANEL, panel, border_radius=10)

        totals = totals or {}
        white = max(0, int(totals.get("white", 0)))
        draws = max(0, int(totals.get("draw", 0)))
        black = max(0, int(totals.get("black", 0)))
        total = white + draws + black

        self.text(f"RESULTADOS ACUMULADOS  •  {total} partidas", x + 12, y + 5, TEXT, self.tiny)
        bar_x = x + 12
        bar_y = y + 21
        bar_w = max(1, w - 24)
        bar_h = 12
        pygame.draw.rect(self.screen, (62, 64, 76), (bar_x, bar_y, bar_w, bar_h), border_radius=4)

        if total:
            values = [white, draws, black]
            colors = [BOARD_LIGHT, (132, 136, 148), (25, 26, 33)]
            segment_widths = [int(bar_w * value / total) for value in values]
            remainder = bar_w - sum(segment_widths)
            segment_widths[max(range(3), key=lambda index: values[index])] += remainder
            cursor = bar_x
            for index, (segment_w, color) in enumerate(zip(segment_widths, colors)):
                if segment_w > 0:
                    rect = pygame.Rect(cursor, bar_y, segment_w, bar_h)
                    pygame.draw.rect(self.screen, color, rect)
                    if index == 2:
                        pygame.draw.rect(self.screen, (205, 207, 215), rect, 1)
                    cursor += segment_w
            pygame.draw.rect(self.screen, (105, 108, 121), (bar_x, bar_y, bar_w, bar_h), 1, border_radius=4)

        label_y = y + 36
        col_w = (w - 24) / 3
        entries = [("B", white, BOARD_LIGHT), ("E", draws, (170, 174, 186)), ("P", black, (205, 207, 215))]
        for index, (label, value, color) in enumerate(entries):
            pct = (100.0 * value / total) if total else 0.0
            text = f"{label} {pct:.1f}% ({value})" if total else f"{label} -- (0)"
            self.text(text, x + 12 + index * col_w, label_y, color, self.tiny)

        recent_results = list(recent_results or [])[-100:]
        recent_total = len(recent_results)
        if recent_total:
            recent_white = recent_results.count("1-0")
            recent_draws = recent_results.count("1/2-1/2")
            recent_black = recent_results.count("0-1")
            recent_label = (
                f"Últimas {recent_total}: "
                f"B {100 * recent_white / recent_total:.0f}%  •  "
                f"E {100 * recent_draws / recent_total:.0f}%  •  "
                f"P {100 * recent_black / recent_total:.0f}%"
            )
        else:
            recent_label = "Últimas 100: aguardando partidas concluídas"
        self.text(recent_label, x + 12, y + 54, MUTED, self.tiny)

    def evolution_status(self, history):
        if len(history) == 0:
            return "Aguardando benchmarks", MUTED
        if len(history) < 4:
            return "Rating em calibração", ACCENT
        recent = [float(item[1]) for item in history[-6:]]
        change = recent[-1] - recent[0]
        if change >= 8:
            return "Aprendendo rápido", ACCENT
        if change >= 1:
            return "Evolução estável", GOLD
        return "Estagnado", RED

    def draw_controls(self, snapshot, x, y, w):
        panel = pygame.Rect(x, y, w, 205)
        pygame.draw.rect(self.screen, PANEL, panel, border_radius=12)
        self.text("CONTROLE DE PARTIDAS", x + 14, y + 12, TEXT, self.small)
        self.text(f"Simultâneas: {self.count}", x + 14, y + 39, GOLD, self.small)

        for label, delta in [("-", -1), ("+", 1)]:
            rect = pygame.Rect(x + 14 + (54 if delta == 1 else 0), y + 67, 42, 32)
            pygame.draw.rect(self.screen, (75, 78, 96), rect, border_radius=7)
            self.text(label, rect.x + 14, rect.y + 5)
            self.buttons.append((rect, "inc" if delta == 1 else "dec"))

        button_y = y + 111
        button_gap = 5
        button_width = (w - 28 - button_gap * 4) // 5
        for i, n in enumerate((1, 10, 25, 50, 100)):
            rect = pygame.Rect(x + 14 + i * (button_width + button_gap), button_y, button_width, 32)
            color = BOARD_DARK if self.count == n else (67, 69, 83)
            pygame.draw.rect(self.screen, color, rect, border_radius=6)
            label = self.small.render(str(n), True, TEXT)
            self.screen.blit(label, label.get_rect(center=rect.center))
            self.buttons.append((rect, n))

        history = snapshot.get("elo_history", [])
        status, color = self.evolution_status(history)
        pygame.draw.circle(self.screen, color, (x + 22, y + 176), 5)
        self.text(status, x + 34, y + 167, color, self.small)

    def draw(self, main_board, snapshot):
        w, h = self.screen.get_size()
        self.screen.fill(BG)
        self.buttons = []
        self.mini_rects = []

        self.text("NEURAL CHESS  /  GENETIC SELF-PLAY", 24, 14, TEXT, self.font)
        self.text(
            f"Geração {snapshot.get('generation', 0)}   •   Melhor fitness {snapshot.get('best_fitness', 0):.1f}",
            24, 42, MUTED, self.small,
        )

        left_x = 24
        top_y = 78
        board_size = max(250, min(410, h - 360, int(w * 0.29)))

        last_move = None
        best_move = None
        try:
            if snapshot.get("last_move"):
                last_move = chess.Move.from_uci(snapshot["last_move"])
            if snapshot.get("best_move"):
                best_move = chess.Move.from_uci(snapshot["best_move"])
        except Exception:
            last_move = best_move = None

        self.board(main_board, left_x, top_y, board_size, mini=False, last_move=last_move, best_move=best_move)

        bar_x = left_x + board_size + 10
        bar_y = top_y
        bar_h = board_size
        bar_w = 12
        pygame.draw.rect(self.screen, (65, 66, 78), (bar_x, bar_y, bar_w, bar_h), border_radius=5)
        val = max(-1, min(1, float(snapshot.get("evaluation", 0))))
        fill = int(bar_h * (val + 1) / 2)
        pygame.draw.rect(self.screen, (230, 230, 236), (bar_x, bar_y + bar_h - fill, bar_w, fill), border_radius=5)

        control_x = left_x
        control_y = top_y + board_size + 18
        control_w = min(470, w - 48)
        self.draw_controls(snapshot, control_x, control_y, control_w)

        graph_y = control_y + 218
        footer_limit = h - 25
        available_h = max(0, footer_limit - graph_y)
        stats_h = 76
        graph_gap = 7
        graph_h = max(70, available_h - stats_h - graph_gap)
        if graph_y + graph_h + graph_gap + stats_h > footer_limit:
            graph_h = max(48, available_h - stats_h - graph_gap)
        stats_y = graph_y + graph_h + graph_gap

        self.draw_graph(snapshot.get("elo_history", []), control_x, graph_y, control_w, graph_h)
        self.draw_result_stats(
            snapshot.get("result_totals", {}),
            snapshot.get("recent_results", []),
            control_x, stats_y, control_w, stats_h,
        )

        grid_x = max(510, int(w * 0.345))
        grid_y = 78
        grid_w = max(200, w - grid_x - 20)
        grid_h = max(200, h - grid_y - 45)
        self.text("PARTIDAS AO VIVO", grid_x, 42, TEXT, self.small)

        boards = snapshot.get("boards", [])[:100]
        count = len(boards)
        if count <= 12:
            cols = 3
        elif count <= 25:
            cols = 5
        elif count <= 50:
            cols = 7
        else:
            cols = 10

        rows = max(1, math.ceil(count / cols))
        cell_w = grid_w / cols
        cell_h = grid_h / rows
        msize = max(28, int(min(cell_w - 10, cell_h - 24, 132)))

        for i, item in enumerate(boards):
            col = i % cols
            row = i // cols
            xx = int(grid_x + col * cell_w)
            yy = int(grid_y + row * cell_h)
            card_w = int(cell_w - 5)
            card_h = int(cell_h - 4)
            card = pygame.Rect(xx, yy, card_w, card_h)
            is_selected = item.get("id") == snapshot.get("focus_id")

            pygame.draw.rect(self.screen, PANEL, card, border_radius=6)
            if is_selected:
                pygame.draw.rect(self.screen, GOLD, card, 2, border_radius=6)

            self.text(f"#{item.get('id', i + 1)} G{item.get('genome', 0) + 1}", xx + 4, yy + 2, MUTED, self.tiny)
            board_x = xx + (card_w - msize) // 2
            board_y = yy + 17
            self.board(item.get("pieces", {}), board_x, board_y, msize, mini=True)

            status = item.get("status", "ativo")
            status_color = ACCENT if status == "ativo" else GOLD
            self.text(status, xx + 4, yy + card_h - 15, status_color, self.tiny)
            self.mini_rects.append((card, item.get("id")))

        footer_y = h - 25
        self.text(
            f"GPU/CPU: {snapshot.get('device', '')}   |   Rating estimado: {snapshot.get('rating', 400):.0f}   |   Loss: {snapshot.get('loss', 0):.4f}",
            24, footer_y, MUTED, self.tiny,
        )
        mode = "TURBO" if snapshot.get("turbo", False) else "ASSISTIR"
        self.text(
            f"Modo: {mode}   SPACE: alternar   S: salvar   ESC: sair",
            max(500, w - 400), footer_y, MUTED, self.tiny,
        )

        pygame.display.flip()
        self.clock.tick(12 if snapshot.get("turbo", False) else 60)

    def events(self):
        changed = None
        quit_ = False
        keys = []
        focus_id = None
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                quit_ = True
            elif e.type == pygame.KEYDOWN:
                keys.append(e.key)
            elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                for rect, val in self.buttons:
                    if rect.collidepoint(e.pos):
                        if val == "dec":
                            self.count = max(1, self.count - 1)
                        elif val == "inc":
                            self.count = min(100, self.count + 1)
                        else:
                            self.count = max(1, min(100, int(val)))
                        changed = self.count
                        break
                if changed is None:
                    for rect, match_id in self.mini_rects:
                        if rect.collidepoint(e.pos):
                            focus_id = match_id
                            self.focus_id = match_id
                            break
        return quit_, keys, changed, focus_id

    def close(self):
        pygame.quit()
import cv2
import numpy as np
import pytesseract
from ultralytics import YOLO

table = {
    "16th note": 0.25,
    "8th note": 0.5,
    "quarter note": 1,
    "half note": 2,
    "dot": "dot"
}

rest = {
    "16th rest": 0.25,
    "8th rest": 0.5,
    "4th rest": 1,  # dataset/model class is named "4th rest", not "quarter rest"
    "half rest": 2,
    "whole rest": 4
}

#———————————————————————————————————————————model setting—————————————————————————————————————————————————————————————————
rhythmModel = YOLO("./runs/detect/rhythm-3/weights/best.pt")
# rhythmModel = YOLO("./runs/detect/train-3/weights/best.pt")
rhythmResults = rhythmModel("./images/test_image/kingGnu6.png", conf=0.4, iou=0)
rhythmResults[0].show()  # display with boxes
rhythmResults[0].save("rhythmOutput.png")  # save annotated image

rhythmBoxes = sorted(rhythmResults[0].boxes, key=lambda b: b.xyxy[0][0].item())

noteModel = YOLO("./runs/detect/fret-2/weights/best.pt")
# noteModel = YOLO("./runs/detect/train/weights/best.pt")
noteResults = noteModel("./images/test_image/kingGnu6.png", conf=0.25, iou=1.0)
noteResults[0].show()  # display with boxes
noteResults[0].save("noteOutput.png")  # save annotated image


#—————————————————————————————————————————Getting the line——————————————————————————————————————————————————————————————————————————————————————

def line_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 215, 255, cv2.THRESH_BINARY_INV)
    height, width = binary.shape

    # use the middle third of the image width to avoid text/annotations on edges
    x_start = width // 3
    x_end = 2 * width // 3
    strip = binary[:, x_start:x_end]
    black_frac = np.sum(strip > 0, axis=1) / (x_end - x_start)

    threshold = 0.30
    min_band_gap = 5
    bands = []
    i = 0
    while i < height:
        if black_frac[i] > threshold:
            start = i
            while i < height and black_frac[i] > threshold:
                i += 1
            center = (start + i) // 2
            if not bands or center - bands[-1] > min_band_gap:
                bands.append(center)
        else:
            i += 1

    print(f"Raw bands detected: {bands}")

    if len(bands) < 2:
        print("Warning: fewer than 2 staff lines detected")
        return bands

    if len(bands) >= 6:
        staff_lines = bands[:6]
    else:
        spacing = bands[1] - bands[0]
        staff_lines = [bands[0] + spacing * n for n in range(6)]

    print("Detected staff lines:", staff_lines)
    return staff_lines



orig = rhythmResults[0].orig_img
arr = line_detection(orig)
for i, y in enumerate(arr):
    print(f"line {i + 1}: y={int(y)}")

debug_lines = orig.copy()
for i, y in enumerate(arr):
    cv2.line(debug_lines, (0, int(y)), (debug_lines.shape[1], int(y)), (0, 0, 255), 1)
    cv2.putText(debug_lines, str(i + 1), (5, int(y) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
cv2.imwrite("debug_staff_lines.png", debug_lines)

#————————————————————————————————————Getting the specific note——————————————————————————————————————————————————————————————————————

def fret_to_note(ycor, staff_lines):
    nearest = min(range(len(staff_lines)), key=lambda i: abs(ycor - staff_lines[i]))
    return nearest + 1

# Standard tuning open string MIDI: E5 B4 G4 D4 A3 E3 (string 1 = top/thinnest)
OPEN_STRINGS = [76, 71, 67, 62, 57, 52]

def ocr_fret_number(image, box):
    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
    pad = 4
    crop = image[max(0, y1-pad):min(image.shape[0], y2+pad),
                 max(0, x1-pad):min(image.shape[1], x2+pad)]
    crop = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    text = pytesseract.image_to_string(thresh, config='--psm 8 -c tessedit_char_whitelist=0123456789Xx/')
    cleaned = ''.join(c for c in text if c.isdigit() or c.lower() == 'x' or c == '/')
    print(f"  OCR raw: {repr(text.strip())}  →  cleaned: '{cleaned}'")

    # A slash between two fret numbers ("5/7") is slide/glissando shorthand —
    # return it as a (from_fret, to_fret) tuple instead of a single int.
    if '/' in cleaned:
        left, _, right = cleaned.partition('/')
        if left.isdigit() and right.isdigit():
            from_fret, to_fret = int(left), int(right)
            if 0 <= from_fret <= 24 and 0 <= to_fret <= 24:
                return (from_fret, to_fret)
        # malformed slash reading — fall through and try to salvage plain digits

    digits = ''.join(c for c in cleaned if c.isdigit())
    has_x = 'x' in cleaned.lower()
    if has_x and not digits:
        return 'x'
    if digits:
        val = int(digits)
        return val if 0 <= val <= 24 else -1
    return -1

#—————————————————————————————————————————Store notes as array——————————————————————————————————————————————————————————————————————————————————————

def _box_area(box):
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    return (x2 - x1) * (y2 - y1)

def dedup_boxes(boxes, iou_thresh=0.5):
    by_area = sorted(boxes, key=_box_area, reverse=True)
    kept = []
    for box in by_area:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        overlaps = False
        for k in kept:
            kx1, ky1, kx2, ky2 = k.xyxy[0].tolist()
            ix1, iy1 = max(x1, kx1), max(y1, ky1)
            ix2, iy2 = min(x2, kx2), min(y2, ky2)
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2 - ix1) * (iy2 - iy1)
                union = (x2-x1)*(y2-y1) + (kx2-kx1)*(ky2-ky1) - inter
                if inter / union > iou_thresh:
                    overlaps = True
                    break
        if not overlaps:
            kept.append(box)
    return kept

note_orig = noteResults[0].orig_img
noteBoxes = sorted(dedup_boxes(noteResults[0].boxes), key=lambda b: b.xyxy[0][0].item())

# Drop any detection that sits directly above a measure line (barline) — those
# are measure-number annotations, not real fret numbers. "Directly above" means
# both: horizontally overlapping that specific measure line's own box (not just
# whichever measure line is nearest by raw x-distance — see the x≈166 bug, where
# a box was closer in x to a later, oddly-cropped measure line than to its own
# and got wrongly dropped), and above the shared staff top line (arr[0], which
# is far less noisy than any individual measure-line detection's y1).
measureLineBoxes = sorted(
    [b for b in rhythmBoxes if rhythmModel.names[int(b.cls[0].item())] == 'measure line'],
    key=lambda b: b.xyxy[0][0].item()
)

def _is_above_measure(box):
    if not arr:
        return False
    bx1, by1, bx2, by2 = box.xyxy[0].tolist()
    if by2 >= arr[0]:
        return False
    return any(bx2 > ml.xyxy[0][0].item() and bx1 < ml.xyxy[0][2].item() for ml in measureLineBoxes)

_before_filter = len(noteBoxes)
noteBoxes = [b for b in noteBoxes if not _is_above_measure(b)]
print(f"Dropped {_before_filter - len(noteBoxes)} box(es) above a measure line (measure numbers)")

noteMidiMap = {}
noteTechMap = {}
_saved_neg1 = False

for box in noteBoxes:
    cls = int(box.cls[0].item())
    name = noteModel.names[cls]
    conf = box.conf[0].item()
    xyxy = box.xyxy[0].tolist()
    center_y = (box.xyxy[0][1].item() + box.xyxy[0][3].item()) / 2
    string_num = fret_to_note(center_y, arr)
    staff_y = int(arr[string_num - 1]) if string_num <= len(arr) else -1
    fret_num = ocr_fret_number(note_orig, box)
    if fret_num == -1:
        fret_num = 1  # OCR couldn't read a digit — default to fret 1 rather than dropping the note
    if fret_num == 'x':
        midi = 'muted'
    elif isinstance(fret_num, tuple):
        midi = OPEN_STRINGS[string_num - 1] + fret_num[0]  # slide's starting pitch
    elif fret_num >= 0:
        midi = OPEN_STRINGS[string_num - 1] + fret_num
    else:
        midi = -1
    noteMidiMap[id(box)] = midi
    noteTechMap[id(box)] = (string_num, fret_num)
    if midi == -1 and not _saved_neg1:
        x1d, y1d, x2d, y2d = [int(v) for v in box.xyxy[0].tolist()]
        pad = 4
        raw_crop = note_orig[max(0, y1d-pad):min(note_orig.shape[0], y2d+pad),
                              max(0, x1d-pad):min(note_orig.shape[1], x2d+pad)]
        big = cv2.resize(raw_crop, None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite("debug_neg1_crop.png", big)
        gray_d = cv2.cvtColor(raw_crop, cv2.COLOR_BGR2GRAY)
        _, thresh_d = cv2.threshold(gray_d, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        cv2.imwrite("debug_neg1_thresh.png", cv2.resize(thresh_d, None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST))
        print(f"  !! Saved debug_neg1_crop.png and debug_neg1_thresh.png for box {xyxy}")
        _saved_neg1 = True

#————————————————————————————————————Group rhythm stems into measures——————————————————————————————————————————————————————————————————————

X_TOL = 10

def _mid_y(box):
    return (box.xyxy[0][1].item() + box.xyxy[0][3].item()) / 2

noteBoxesNoTie = [box for box in noteBoxes
                  if 'tie' not in noteModel.names[int(box.cls[0].item())].lower()]
tieBoxes       = [box for box in noteBoxes
                  if 'tie' in noteModel.names[int(box.cls[0].item())].lower()]

# Every measure line (barline) starts a fresh measure and a fresh window for
# matching fret boxes to stems: everything before the first measure line is
# measure 0, everything between measure lines N-1 and N is measure N, etc.
# Notes are only matched against stems in the same window, so a fret box can't
# leak across a barline into the wrong measure.
measure_bounds = [float('-inf')] + [b.xyxy[0][0].item() for b in measureLineBoxes] + [float('inf')]

measures = []             # list[ list[ (rhythmBox, entries) ] ], one inner list per measure
measureNoteBoxes = []      # list[ list[noteBox] ], fret-number boxes in each measure (parallel to `measures`)
entriesByRhythmBox = {}   # id(rhythmBox) -> entries, used by tie resolution below

for lo, hi in zip(measure_bounds, measure_bounds[1:]):
    stems_in_measure = [b for b in rhythmBoxes
                         if rhythmModel.names[int(b.cls[0].item())] != 'measure line'
                         and lo <= b.xyxy[0][0].item() < hi]
    if not stems_in_measure:
        continue

    notes_in_measure = [b for b in noteBoxesNoTie if lo <= b.xyxy[0][0].item() < hi]

    measure_entries = []
    for rhythmBox in stems_in_measure:
        rhythm_name = rhythmModel.names[int(rhythmBox.cls[0].item())]
        if rhythm_name in rest:
            entries = "rest"
        else:
            stem_x = rhythmBox.xyxy[0][0].item()
            # Keep at most one note per string per chord — a real chord can't have
            # two different frets on the same string at once, and writing such a
            # duplicate straight into a Guitar Pro beat corrupts the gp5 binary
            # (verified: two Notes on the same string in one beat makes the file
            # unparseable; two different strings round-trip fine). When more than
            # one candidate box lands on the same string within X_TOL, keep
            # whichever is closest in x to the stem.
            by_string = {}
            for box in notes_in_measure:
                note_x = box.xyxy[0][0].item()
                dist = abs(note_x - stem_x)
                if dist <= X_TOL:
                    string_num, fret_num = noteTechMap.get(id(box), (None, None))
                    if string_num is None:
                        continue
                    best = by_string.get(string_num)
                    if best is None or dist < best[0]:
                        by_string[string_num] = (dist, fret_num)
            entries = [[(string_num, fret_num), "na"] for string_num, (_, fret_num) in by_string.items()]
            if not entries:
                # A note-type stem (not a rest) with nothing matched to it almost
                # certainly means the fret-number detection was missed, not that
                # the beat is silent — default to open string 3 rather than
                # letting it fall through as an empty/rest beat downstream.
                entries.append([(3, 0), "na"])
        entriesByRhythmBox[id(rhythmBox)] = entries
        measure_entries.append((rhythmBox, entries))

    measures.append(measure_entries)
    measureNoteBoxes.append(notes_in_measure)

print(f"Detected {len(measureLineBoxes)} measure line(s) → {len(measures)} measure(s)")

#————————————————————————————————————Process ties——————————————————————————————————————————————————————————————————————

rhythm_list = [rb for measure_entries in measures for rb, _ in measure_entries]

def note_y(entry):
    string_num = entry[0][0]
    if string_num is None or string_num < 1 or string_num > len(arr):
        return float('inf')
    return arr[string_num - 1]

print(f"Total ties found: {len(tieBoxes)}")
for tie_box in tieBoxes:
    tx1, ty1, tx2, ty2 = tie_box.xyxy[0].tolist()
    tie_mid_y = (ty1 + ty2) / 2

    left_candidates = [b for b in rhythm_list if b.xyxy[0][0].item() <= tx1]
    left_stem  = max(left_candidates, key=lambda b: b.xyxy[0][0].item()) if left_candidates else min(rhythm_list, key=lambda b: abs(b.xyxy[0][0].item() - tx1))
    right_stem = min((b for b in rhythm_list if b is not left_stem), key=lambda b: abs(b.xyxy[0][0].item() - tx2))

    left_notes  = entriesByRhythmBox[id(left_stem)]  if isinstance(entriesByRhythmBox[id(left_stem)],  list) else []
    right_notes = entriesByRhythmBox[id(right_stem)] if isinstance(entriesByRhythmBox[id(right_stem)], list) else []

    left_idx  = min(range(len(left_notes)),  key=lambda i: abs(note_y(left_notes[i])  - tie_mid_y)) if left_notes  else None
    right_idx = min(range(len(right_notes)), key=lambda i: abs(note_y(right_notes[i]) - tie_mid_y)) if right_notes else None

    if left_idx is not None:
        if right_idx is not None:
            left_notes[left_idx][1]   = "start"
            right_notes[right_idx][1] = "end"
        else:
            left_notes[left_idx][1] = "let-ring"
    # left empty, right has note → misdetection, leave both "na"

for m_idx, (measure_entries, note_boxes) in enumerate(zip(measures, measureNoteBoxes)):
    print(f"\n=== Measure {m_idx + 1} ===")

    print("  Rhythm:")
    for rhythmBox, notes in measure_entries:
        rhythm_name = rhythmModel.names[int(rhythmBox.cls[0].item())]
        box_xyxy = [round(v, 1) for v in rhythmBox.xyxy[0].tolist()]
        print(f"    {rhythm_name:12s} box={box_xyxy}  notes={notes}")

    print("  Fret numbers:")
    if not note_boxes:
        print("    (none)")
    for box in note_boxes:
        string_num, fret_num = noteTechMap.get(id(box), (None, None))
        box_xyxy = [round(v, 1) for v in box.xyxy[0].tolist()]
        print(f"    string {string_num}, fret {fret_num}  box={box_xyxy}")

#── Validate rhythm against the time signature ──────────────────────────────
#
# Both exporters below (Guitar Pro and MusicXML) need to agree on which
# detections actually fit in each measure, so that check happens exactly once
# here and both exporters build from the same validated data — instead of each
# reimplementing (and potentially drifting from) the same logic.

# Time signature — hardcoded for now; eventually this should be asked of the user.
TIME_SIG_NUMERATOR = 4
TIME_SIG_DENOMINATOR = 4
MEASURE_CAPACITY_QL = TIME_SIG_NUMERATOR * (4.0 / TIME_SIG_DENOMINATOR)
_QL_EPS = 1e-6

class _Scalar:
    """Mimics the `.item()` accessor on a YOLO box's `.cls` tensor entry."""
    __slots__ = ('_v',)
    def __init__(self, v):
        self._v = v
    def item(self):
        return self._v

class _FillerRestBox:
    """Stand-in for a detected rhythm box, used only for rest(s) synthesized to
    pad out an underfilled measure. Supports just the `.cls` accessor that the
    validator/exporters read — it was never actually detected in the image."""
    def __init__(self, cls_id):
        self.cls = [_Scalar(cls_id)]

_rest_name_to_cls = {name: cls_id for cls_id, name in rhythmModel.names.items() if name in rest}
_REST_QLS_DESC = sorted(rest.values(), reverse=True)  # e.g. [4.0, 2.0, 1.0, 0.5, 0.25]
_REST_QL_TO_NAME = {ql: name for name, ql in rest.items()}

def _filler_rests(fill_ql, eps=1e-6):
    """Greedily decompose `fill_ql` into standard rest lengths (largest first,
    same as writing 'change' for a partial measure) and return the matching
    (fillerBox, "rest") entries — same shape as a normally-detected rest entry."""
    entries = []
    for rest_ql in _REST_QLS_DESC:
        while fill_ql >= rest_ql - eps:
            cls_id = _rest_name_to_cls.get(_REST_QL_TO_NAME[rest_ql])
            if cls_id is None:
                break
            entries.append((_FillerRestBox(cls_id), "rest"))
            fill_ql -= rest_ql
    return entries

def validate_measures(measures, capacity_ql, eps=1e-6):
    """Trim each measure's detections to fit the time signature.

    Sanity check against the time signature: track how much of the measure is
    still unaccounted for as we place detections, in detection order.
      1. If every detection in the subarray gets placed before space runs out,
         the subarray is "depleted" — move on to the next measure normally.
      2. If space runs out (exactly, or the next detection simply doesn't fit)
         while detections remain, that's a rhythm mismatch (overflow): the
         extra detections are dropped for THIS measure rather than forced in
         or spilled into the next one.
      3. If the subarray runs out of detections while space remains, that's a
         mismatch too (underflow): flagged, but nothing is borrowed from the
         next subarray to fill the gap.
    Either way, one bad measure's detections can't drag every later measure's
    beat position out of alignment with the rest of the piece.
    """
    trimmed = []
    mismatch_count = 0

    for m_idx, measure_entries in enumerate(measures):
        remaining_ql = capacity_ql
        last_ql = None
        kept = []

        for entry_idx, (rhythmBox, notes) in enumerate(measure_entries):
            rhythm_name = rhythmModel.names[int(rhythmBox.cls[0].item())]

            if rhythm_name == 'dot':
                added = 0.5 * last_ql if last_ql is not None else None
                if not kept or added is None or added > remaining_ql + eps:
                    mismatch_count += 1
                    print(f"  !! Measure {m_idx + 1}: dot doesn't fit in the remaining "
                          f"{remaining_ql:.3f} — dropping the rest of this measure's "
                          f"detections ({len(measure_entries) - entry_idx} left) and "
                          f"moving to the next measure.")
                    break
                kept.append((rhythmBox, notes))
                remaining_ql -= added
                if remaining_ql <= eps:
                    break
                continue

            if notes == 'rest':
                ql = rest.get(rhythm_name)
            else:
                ql = table.get(rhythm_name)
                if ql == 'dot':
                    ql = None
            if ql is None:
                continue

            if ql > remaining_ql + eps:
                mismatch_count += 1
                print(f"  !! Measure {m_idx + 1}: '{rhythm_name}' (length {ql}) doesn't fit "
                      f"in the remaining {remaining_ql:.3f} — dropping the rest of this "
                      f"measure's detections ({len(measure_entries) - entry_idx} left) and "
                      f"moving to the next measure.")
                break

            kept.append((rhythmBox, notes))
            last_ql = ql
            remaining_ql -= ql
            if remaining_ql <= eps:
                break
        else:
            if remaining_ql > eps:
                mismatch_count += 1
                fillers = _filler_rests(remaining_ql, eps)
                print(f"  !! Measure {m_idx + 1}: ran out of detections with {remaining_ql:.3f} "
                      f"of the measure still unfilled (underflow) — padding with "
                      f"{len(fillers)} rest(s) and starting a new measure from the next "
                      f"subarray rather than borrowing from it.")
                kept.extend(fillers)

        trimmed.append(kept)

    return trimmed, mismatch_count

validated_measures, mismatch_count = validate_measures(
    measures if measures else [[]], MEASURE_CAPACITY_QL, _QL_EPS
)
print(f"Rhythm validation: {mismatch_count} measure(s) had a mismatch against "
      f"{TIME_SIG_NUMERATOR}/{TIME_SIG_DENOMINATOR} — see warnings above")

#── Guitar Pro output ────────────────────────────────────────────────────────

import guitarpro

gp_dur_map = {0.25: 16, 0.5: 8, 1.0: 4, 2.0: 2, 4.0: 1}

song = guitarpro.Song()
song.measureHeaders = [guitarpro.MeasureHeader() for _ in validated_measures]
for header in song.measureHeaders:
    header.timeSignature.numerator = TIME_SIG_NUMERATOR
    header.timeSignature.denominator.value = TIME_SIG_DENOMINATOR

track = guitarpro.Track(song, number=1, strings=[
    guitarpro.GuitarString(i + 1, OPEN_STRINGS[i]) for i in range(6)
])
song.tracks = [track]

track.measures = []
for header, measure_entries in zip(song.measureHeaders, validated_measures):
    gp_measure = guitarpro.Measure(track, header)
    voice = gp_measure.voices[0]
    last_beat = None

    for rhythmBox, notes in measure_entries:
        rhythm_name = rhythmModel.names[int(rhythmBox.cls[0].item())]

        if rhythm_name == 'dot':
            if last_beat is not None:
                last_beat.duration.isDotted = True
            continue

        if notes == 'rest':
            ql = rest.get(rhythm_name)
            if ql is None:
                continue
            beat = guitarpro.Beat(voice, duration=guitarpro.Duration(value=gp_dur_map.get(ql, 4)),
                                   status=guitarpro.BeatStatus.rest)
            voice.beats.append(beat)
            last_beat = beat
            continue

        ql = table.get(rhythm_name)
        if ql is None or ql == 'dot':
            continue

        beat = guitarpro.Beat(voice, duration=guitarpro.Duration(value=gp_dur_map.get(ql, 4)))

        if len(notes) == 0:
            beat.status = guitarpro.BeatStatus.rest
        else:
            beat.status = guitarpro.BeatStatus.normal
            for entry in notes:
                string_num, fret_num = entry[0]
                tie_type = entry[1]
                if string_num is None:
                    continue
                if fret_num == 'x':
                    gp_note = guitarpro.Note(beat, value=0, string=string_num,
                                             type=guitarpro.NoteType.dead)
                elif isinstance(fret_num, tuple):
                    # "5/7" slash reading — a slide/glissando starting at fret 5.
                    # There's no separate detected note to slide *to* here (one
                    # OCR'd box, one beat), so this is modeled as a slide out of
                    # this note (direction from whether the target is higher or
                    # lower) rather than inventing a second note at the target fret.
                    from_fret, to_fret = fret_num
                    gp_note = guitarpro.Note(beat, value=from_fret, string=string_num,
                                             type=guitarpro.NoteType.normal)
                    gp_note.effect.slides = [
                        guitarpro.SlideType.outUpwards if to_fret > from_fret
                        else guitarpro.SlideType.outDownwards
                    ]
                elif isinstance(fret_num, int) and fret_num >= 0:
                    gp_note = guitarpro.Note(beat, value=fret_num, string=string_num,
                                             type=guitarpro.NoteType.normal)
                    if tie_type == 'end':
                        gp_note.type = guitarpro.NoteType.tie
                else:
                    continue
                beat.notes.append(gp_note)

        voice.beats.append(beat)
        last_beat = beat

    track.measures.append(gp_measure)

guitarpro.write(song, 'output.gp5')
print("Saved output.gp5")

#── music21 (MusicXML) output ───────────────────────────────────────────────

import music21 as m21

score = m21.stream.Score()
part = m21.stream.Part()
part.partName = "Guitar"
part.append(m21.instrument.Guitar())

for m_idx, measure_entries in enumerate(validated_measures):
    m21_measure = m21.stream.Measure(number=m_idx + 1)
    if m_idx == 0:
        m21_measure.append(m21.meter.TimeSignature(f'{TIME_SIG_NUMERATOR}/{TIME_SIG_DENOMINATOR}'))
    last_element = None

    for rhythmBox, notes in measure_entries:
        rhythm_name = rhythmModel.names[int(rhythmBox.cls[0].item())]

        if rhythm_name == 'dot':
            if last_element is not None:
                last_element.duration.dots = 1
            continue

        if notes == 'rest':
            ql = rest.get(rhythm_name)
            if ql is None:
                continue
            element = m21.note.Rest(quarterLength=ql)
            m21_measure.append(element)
            last_element = element
            continue

        ql = table.get(rhythm_name)
        if ql is None or ql == 'dot':
            continue

        built_notes = []
        for entry in notes:
            string_num, fret_num = entry[0]
            tie_type = entry[1]
            if string_num is None:
                continue
            if fret_num == 'x':
                gp_note = m21.note.Note(OPEN_STRINGS[string_num - 1], quarterLength=ql)
                gp_note.notehead = 'x'
                display_fret = 0
            elif isinstance(fret_num, tuple):
                # "5/7" slide reading — music21 gets a plain note at the starting
                # fret; the slide effect itself is only modeled on the Guitar Pro
                # side for now.
                from_fret, _to_fret = fret_num
                gp_note = m21.note.Note(OPEN_STRINGS[string_num - 1] + from_fret, quarterLength=ql)
                display_fret = from_fret
            elif isinstance(fret_num, int) and fret_num >= 0:
                gp_note = m21.note.Note(OPEN_STRINGS[string_num - 1] + fret_num, quarterLength=ql)
                display_fret = fret_num
            else:
                continue
            gp_note.articulations.append(m21.articulations.StringIndication(string_num))
            gp_note.articulations.append(m21.articulations.FretIndication(display_fret))
            if tie_type == 'start':
                gp_note.tie = m21.tie.Tie('start')
            elif tie_type == 'end':
                gp_note.tie = m21.tie.Tie('stop')
            built_notes.append(gp_note)

        if not built_notes:
            element = m21.note.Rest(quarterLength=ql)
        elif len(built_notes) == 1:
            element = built_notes[0]
        else:
            element = m21.chord.Chord(built_notes, quarterLength=ql)

        m21_measure.append(element)
        last_element = element

    part.append(m21_measure)

score.append(part)
score.write('musicxml', fp='output.xml')
print("Saved output.xml")


from cStringIO import StringIO
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter, PDFPageAggregator
from pdfminer.layout import LAParams, LTTextBoxHorizontal, LTTextLineHorizontal, LTLine
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfparser import PDFParser
import re
import math
import pandas as pd
from warnings import warn


def get_layouts(fname):
    cstr = StringIO()
    with open(fname, 'rb') as fp:
        cstr.write(fp.read())
    cstr.seek(0)
    doc = PDFDocument(PDFParser(cstr))
    rsrcmgr = PDFResourceManager()
    laparams = LAParams(line_margin=0.000001, char_margin=1)
    device = PDFPageAggregator(rsrcmgr, laparams=laparams)
    interpreter = PDFPageInterpreter(rsrcmgr, device)

    layouts = list()
    for page in PDFPage.create_pages(doc):
        interpreter.process_page(page)
        layout = device.get_result()
        layouts.append(layout)

    return layouts


def get_pages_content(layouts, header_re, footer_re, v_margin=1):
    pages = list()
    for layout in layouts:
        objs = layout._objs
        footer_matches = find_text(objs, footer_re)
        footer_obj = get_bottom_most(footer_matches)

        header_matches = find_text(objs, header_re)
        header_obj = get_top_most(header_matches)

        # Get min & max x values
        x0 = min(map(lambda x: x.bbox[0], objs))
        x1 = max(map(lambda x: x.bbox[2], objs))
        y0 = footer_obj.bbox[3]
        y1 = header_obj.bbox[1]

        bbox = (x0, y0 + v_margin, x1, y1 - v_margin)
        objs = get_objs_in_bound(objs, bbox)
        pages.append(objs)
    return pages


def find_text(objs, regexp):
    p = re.compile(regexp)

    objs_fnd = list()
    for obj in objs:
        try:
            text = obj.get_text()
            m = p.search(text)
            if m:
                objs_fnd.append(obj)
        except (AttributeError, TypeError):
            pass
    return objs_fnd


def pt_in_bbox(pt, bbox):
    x_in = bbox[0] <= pt[0] <= bbox[2]
    y_in = bbox[1] <= pt[1] <= bbox[3]
    return x_in and y_in


def get_objs_in_bound(objs, bbox, types=None, partial=True):
    ans = list()
    for obj in objs:
        if (types is None) or (type(obj) in types):
            pts = [(obj.x0, obj.y0), (obj.x0, obj.y1),
                   (obj.x1, obj.y0), (obj.x1, obj.y1)]
            pts_in = map(lambda pt: pt_in_bbox(pt, bbox), pts)
            is_in = any(pts_in) if partial else all(pts_in)
            if is_in:
                ans.append(obj)
    return ans


def union_bbox(bbox1, bbox2):
    x0 = min(bbox1[0], bbox2[0])
    x1 = max(bbox1[2], bbox2[2])
    y0 = min(bbox1[1], bbox2[1])
    y1 = max(bbox1[3], bbox2[3])
    return x0, y0, x1, y1


def get_top_most(objs):
    """Get the topmost obj, using leftmost to break ties
    """
    top_obj = objs[0]
    for obj in objs:
        if obj.y1 > top_obj.y1:
            top_obj = obj
        elif (obj.y1 == top_obj.y1) and (obj.x0 < top_obj.x0):
            top_obj = obj
    return top_obj


def get_bottom_most(objs):
    """Get the bottommost obj, using rightmost to break ties
    """
    bottom_obj = objs[0]
    for obj in objs:
        if obj.y0 < bottom_obj.y0:
            bottom_obj = obj
        elif (obj.y0 == bottom_obj.y0) and (obj.x1 > bottom_obj.x1):
            bottom_obj = obj
    return bottom_obj


def get_text_lines(objs):
    lines = list()
    for obj in objs:
        if type(obj) == LTTextLineHorizontal:
            lines.append(obj)
        elif type(obj) == LTTextBoxHorizontal:
            objs_sub = get_text_lines(obj._objs)
            map(lines.append, objs_sub)

    return lines


def get_table(objs, v_margin=5.0):
    """ Heuristically find columns/rows positions based on non-overlapping positions
    :param objs:
    :param v_margin: Fudge factor to account for vertical whitespace
    :return:
    """
    # Filter out non-text boxes
    objs = get_text_lines(objs)
    if len(objs) == 0:
        return None

    # Sort objects by leftmost coord, and figure out the column bounds
    objs = sorted(objs, key=lambda a: a.x0)
    col_num = 0
    cols = dict()
    bound = [objs[0].x0, objs[0].x1]
    for obj in objs:
        if obj.x0 <= bound[1]:
            # Left is within the bounds...expand it
            bound[1] = max(bound[1], obj.x1)
        else:
            # Start a new column
            col_num += 1
            bound = [obj.x0, obj.x1]
        cols[obj] = col_num

    # Sort objects from topmost to bottommost, and get the row bounds
    objs = sorted(objs, key=lambda a: a.y0)

    row_num = 0
    rows = dict()
    bound = [objs[0].y0, objs[0].y1 - v_margin]
    for obj in objs:
        if obj.y0 <= bound[1]:
            bound[1] = max(bound[1], obj.y1 - v_margin)
        else:
            row_num += 1
            bound = [obj.y0, obj.y1 - v_margin]
        rows[obj] = row_num

    # Renumber rows so in ascending order so top row=0
    max_row = max([rows[obj] for obj in rows])
    max_col = max([cols[obj] for obj in cols])
    for obj in rows:
        rows[obj] = (rows[obj] - max_row) * -1

    table = map(lambda x: [None]*(max_col+1), range(0,max_row+1))
    for obj in objs:
        (r, c) = (rows[obj], cols[obj])
        table[r][c] = obj.get_text().replace("\n", " ").strip()

    df = pd.DataFrame(table[1:len(table)], columns=table[0])
    return df


def get_connected_lines(objs, epsilon=0.0):
    objs = filter(lambda x: type(x) == LTLine, objs)
    vertices = list()
    for obj in objs:
        vertices.append((obj.x0, obj.y0, obj))
        vertices.append((obj.x1, obj.y1, obj))

    dists = list()
    for i in range(len(vertices)):
        for j in range(i+1, len(vertices)):
            a = vertices[i]
            b = vertices[j]
            dist = math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
            dists.append((dist, a[2], b[2]))

    dists = filter(lambda x: x[0] <= epsilon, dists)

    chains = dict()
    for dist in dists:
        a = dist[1]
        b = dist[2]
        if (a not in chains) and (b not in chains):
            chain = [a, b]
            chains[a] = chain
            chains[b] = chain
        elif (a in chains) and (b not in chains):
            chain = chains[a]
            chain.append(b)
            chains[b] = chain
        elif (a not in chains) and (b in chains):
            chain = chains[b]
            chain.append(a)
            chains[a] = chain
        elif chains[a] != chains[b]:
            chain = chains[a]
            chains_b = chains[b]
            for obj in chains_b:
                chain.append(obj)
                chains[obj] = chain

    polygons = chains.values()
    ids = {id(x): x for x in polygons}

    return ids.values()


def get_bbox(objs):
    bbox = objs[0].bbox
    for obj in objs:
        bbox = (min(obj.bbox[0], bbox[0]),
                min(obj.bbox[1], bbox[1]),
                max(obj.bbox[2], bbox[2]),
                max(obj.bbox[3], bbox[3]))

    return bbox


def get_underlined_text(pages, v_margin=2, h_margin=5, line_width_max=1):
    ans = list()
    for i in range(len(pages)):
        text_objs = filter(lambda x: type(x) == LTTextBoxHorizontal, pages[i])
        line_objs = filter(lambda x: type(x) == LTLine, pages[i])

        for text in text_objs:
            underlined = False

            for line in line_objs:
                x_matched = (text.bbox[0] >= (line.bbox[0]-h_margin)) and (text.bbox[2] <= (line.bbox[2]+h_margin))
                y_matched = math.fabs(text.bbox[1] - line.bbox[3]) <= v_margin
                width_match = line.linewidth <= line_width_max
                underlined |= (x_matched and y_matched and width_match)

            if underlined:
                ans.append({"Page": i, "Text": text.get_text().rstrip(), "BBox": text.bbox})

    return ans


def get_text_in_boxes(pages, epsilon=1.1):
    """Find text that is surrounded by lines (i.e., text in a box)
    """

    headers = list()
    for i in range(len(pages)):
        page_headers = list()
        polygons = get_connected_lines(pages[i], epsilon=epsilon)
        bboxes = map(get_bbox, polygons)
        bboxes = sorted(bboxes, key=lambda x: -1 * x[1])  # Sort by y

        for j in range(len(bboxes)):
            objs = get_objs_in_bound(pages[i], bboxes[j], types=set([LTTextBoxHorizontal]))
            objs = filter(lambda x: x, objs)
            objs = filter(lambda x: len(x.get_text().strip()) > 0, objs)
            if len(objs) == 1:
                title = objs[0].get_text().rstrip()
                bbox = union_bbox(bboxes[j], get_bbox(objs))
                page_headers.append({"Page": i, "Text": title, "BBox": bbox})
            elif len(objs) > 1:
                warn("Ignoring unexpected number of text objects in section header page {} - {}".format(i, objs))

        # Sort the headers from topmost to bottomost and then add to list
        page_headers = sorted(page_headers, key=lambda x: -1*x["BBox"][1])
        map(headers.append, page_headers)
    return headers


def sectionize(pages, headers):
    """Give a list of header objects, get the objects between each header
    :param headers:
    :return:
    """
    # Get the objs between each text box
    epsilon = 1
    sections = list()
    for i in range(len(headers)):
        header = headers[i]

        # Info for start of section
        page_start = header["Page"]
        y1_start = header["BBox"][1]

        # Find end of section
        if i == len(headers) - 1:  # Last header, default to last page
            page_end = len(pages) - 1
            y0_end = get_bbox(pages[page_end])[1]
        else:
            page_end = headers[i+1]["Page"]
            y0_end = headers[i+1]["BBox"][3]

        # Enumerate all pages in between the text boxes
        page_objs = list()
        for j in range(page_start, page_end+1):
            bbox = get_bbox(pages[j])
            (x0, x1) = (bbox[0], bbox[2])

            if j == page_start:
                y1 = y1_start
            else:
                y1 = bbox[3]

            if j == page_end:
                y0 = y0_end
            else:
                y0 = bbox[1]

            bbox = (x0, y0+epsilon, x1, y1-epsilon)
            objs = get_objs_in_bound(pages[j], bbox)
            page_objs.append(objs)

        sections.append({"Text": header["Text"], "Pages": page_objs})

    return sections


def print_sections(sections, types=None):
    for section in sections:
        print "[{}]".format(section["Text"])
        for i in range(len(section["Pages"])):
            print "\t[Page {}]".format(i)
            for obj in section["Pages"][i]:
                if (types is not None) and (type(obj) in types):
                    print "\t{}".format(obj)
            print ""
        print ""


def single_val(l):
    if len(l) == 1:
        return l[0]
    elif len(l) == 0:
        return None
    else:
        raise Exception('More than 1 unique value passed to single')


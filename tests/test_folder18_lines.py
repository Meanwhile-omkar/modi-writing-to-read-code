import numpy as np

from modi.segmentation.folder18_lines import remove_edge_ink


def test_ink_touching_the_page_edge_is_removed_and_writing_is_kept():
    page = np.full((300, 400), 255, np.uint8)
    page[0:12, 30:370] = 0       # dark scan border along the top edge
    page[290:300, :] = 0         # bar along the bottom
    page[100:110, 50:350] = 0    # a line of writing, away from the edges
    clean = remove_edge_ink(page)
    assert (clean[0:12] == 255).all() and (clean[290:] == 255).all()
    assert (clean[100:110, 50:350] == 0).all()

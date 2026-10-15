from modi.evaluation.folder18_gt import is_body, join, letters


def test_strict_letters_are_the_handouts_14_printed_pages():
    printed = {n: t for n, (_, t) in letters().items()}
    assert printed == {3: [14], 5: [20], 8: [40], 11: [50, 52, 54, 56], 12: [58, 60, 62, 64], 13: [66, 68], 15: [76]}
    assert all(f == [p + 1 for p in t] for f, t in letters().values())  # handwriting is the page after its print


def test_non_body_lines_are_dropped():
    for line in ['No. 3]', 'RADHABAI', 'क्र. ३] [७-५-१७३३', '१२', '† टीप', '* * *', '']:
        assert not is_body(line), line
    for line in ['श्री', 'श्री यासह चिरंजीव विजीभव राजमान्य राज', 'ष्ट उत्तमच जाली.']:
        assert is_body(line), line


def test_join_follows_the_line_end_rule():
    assert join(['# comment', 'श्री यासह राज+', 'श्री वाजीराउ', '', 'यांसी']) == 'श्री यासह राजश्री वाजीराउ यांसी'


def test_only_an_explicit_status_line_marks_a_letter_proofread():
    from modi.evaluation.folder18_gt import NOT_DONE, is_proofread
    assert not is_proofread(NOT_DONE + '\nश्री')
    assert is_proofread('# Letter 3\n# status: proofread\nश्री')
    assert is_proofread('# status: PROOFREAD  <- change to "# status: proofread" when every line matches the page')

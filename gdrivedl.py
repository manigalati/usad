#!/usr/bin/env python
from __future__ import unicode_literals
import json
import os
import re
import sys
import unicodedata

try:
    #Python3
    from urllib.request import Request, urlopen
except ImportError:
    #Python2
    from urllib2 import Request, urlopen

ITEM_URL = 'https://drive.google.com/open?id={id}'
FILE_URL = 'https://docs.google.com/uc?export=download&id={id}&confirm={confirm}'
FOLDER_URL = 'https://drive.google.com/drive/folders/{id}'

ID_PATTERNS = [
    re.compile('/file/d/([0-9A-Za-z_-]{10,})(?:/|$)', re.IGNORECASE),
    re.compile('id=([0-9A-Za-z_-]{10,})(?:&|$)', re.IGNORECASE),
    re.compile('([0-9A-Za-z_-]{10,})', re.IGNORECASE)
]
FILE_PATTERN = re.compile("itemJson: (\[.*?)};</script>",
                          re.DOTALL | re.IGNORECASE)
FOLDER_PATTERN = re.compile("window\['_DRIVE_ivd'\] = '(.*?)';",
                            re.DOTALL | re.IGNORECASE)
CONFIRM_PATTERN = re.compile("download_warning[0-9A-Za-z_-]+=([0-9A-Za-z_-]+);",
                             re.IGNORECASE)
FOLDER_TYPE = 'application/vnd.google-apps.folder'

def output(text):
    try:
        sys.stdout.write(text)
    except UnicodeEncodeError:
        sys.stdout.write(text.encode('utf8'))

# Big thanks to leo_wallentin for below sanitize function (modified slightly for this script)
# https://gitlab.com/jplusplus/sanitize-filename/-/blob/master/sanitize_filename/sanitize_filename.py
def sanitize(filename):
    blacklist = ["\\", "/", ":", "*", "?", "\"", "<", ">", "|", "\0"]
    reserved = [
        "CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5",
        "COM6", "COM7", "COM8", "COM9", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5",
        "LPT6", "LPT7", "LPT8", "LPT9",
    ]

    filename = "".join(c for c in filename if c not in blacklist)
    filename = "".join(c for c in filename if 31 < ord(c))
    filename = unicodedata.normalize("NFKD", filename)
    filename = filename.rstrip(". ")
    filename = filename.strip()

    if all([x == "." for x in filename]):
        filename = "_" + filename
    if filename in reserved:
        filename = "_" + filename
    if len(filename) == 0:
        filename = "_"
    if len(filename) > 255:
        parts = re.split(r"/|\\", filename)[-1].split(".")
        if len(parts) > 1:
            ext = "." + parts.pop()
            filename = filename[:-len(ext)]
        else:
            ext = ""
        if filename == "":
            filename = "_"
        if len(ext) > 254:
            ext = ext[254:]
        maxl = 255 - len(ext)
        filename = filename[:maxl]
        filename = filename + ext
        filename = filename.rstrip(". ")
        if len(filename) == 0:
            filename = "_"

    return filename


def process_item(id, directory):
    url = ITEM_URL.format(id=id)
    resp = urlopen(url)
    url = resp.geturl()
    html = resp.read().decode('utf-8')

    if '/file/' in url:
        match = FILE_PATTERN.search(html)
        data = match.group(1).replace('\/', '/')
        data = data.replace(r'\x5b', '[').replace(r'\x22', '"').replace(r'\x5d', ']').replace(r'\n','')
        data = json.loads(data)

        file_name = sanitize(data[1])
        file_size = int(data[25][2])
        file_path = os.path.join(directory, file_name)

        process_file(id, file_path, file_size)
    elif '/folders/' in url:
        process_folder(id, directory, html=html)
    elif 'ServiceLogin' in url:
        sys.stderr.write('Id {} does not have link sharing enabled'.format(id))
        sys.exit(1)
    else:
        sys.stderr.write('That id {} returned an unknown url'.format(id))
        sys.exit(1)


def process_folder(id, directory, html=None):
    if not html:
        url = FOLDER_URL.format(id=id)
        html = urlopen(url).read().decode('utf-8')

    match = FOLDER_PATTERN.search(html)
    data = match.group(1).replace('\/', '/')
    data = data.replace(r'\x5b', '[').replace(r'\x22', '"').replace(r'\x5d', ']').replace(r'\n','')
    data = json.loads(data)

    if not os.path.exists(directory):
        os.mkdir(directory)
        output('Directory: {directory} [Created]\n'.format(directory=directory))
    else:
        output('Directory: {directory} [Exists]\n'.format(directory=directory))

    if not data[0]:
        return

    for item in sorted(data[0], key=lambda i: i[3] == FOLDER_TYPE):
        item_id = item[0]
        item_name = sanitize(item[2])
        item_type = item[3]
        item_size = item[13]
        item_path = os.path.join(directory, item_name)

        if item_type == FOLDER_TYPE:
            process_folder(item_id, item_path)
        else:
            process_file(item_id, item_path, int(item_size))


def process_file(id, file_path, file_size, confirm='', cookies=''):
    if os.path.exists(file_path):
        output('{file_path} [Exists]\n'.format(file_path=file_path))
        return

    url = FILE_URL.format(id=id, confirm=confirm)
    req = Request(url, headers={'Cookie': cookies,
                                'User-Agent': 'Mozilla/5.0'})
    resp = urlopen(req)
    cookies = resp.headers.get('Set-Cookie') or ''

    if not confirm and 'download_warning' in cookies:
        confirm = CONFIRM_PATTERN.search(cookies)
        return process_file(id, file_path, file_size, confirm.group(1), cookies)

    output(file_path + '\n')

    try:
        with open(file_path, 'wb') as f:
            dl = 0
            while True:
                chunk = resp.read(4096)
                if not chunk:
                    break

                if b'Too many users have viewed or downloaded this file recently' in chunk:
                    raise Exception('Quota exceeded for this file')

                dl += len(chunk)
                f.write(chunk)
                done = int(50 * dl / file_size)
                output("\r[{}{}] {:.2f}MB/{:.2f}MB".format(
                    '=' * done,
                    ' ' *
                    (50 - done),
                    dl / 1024 / 1024,
                    file_size / 1024 / 1024
                ))
                sys.stdout.flush()
    except:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise

    output('\n')


def get_arg(pos, default=None):
    try:
        return sys.argv[pos]
    except IndexError:
        return default


if __name__ == '__main__':
    url = get_arg(1, '').strip()
    directory = get_arg(2, './').strip()
    id = ''

    if not url:
        sys.stderr.write('A Google Drive URL is required')
        sys.exit(1)

    for pattern in ID_PATTERNS:
        match = pattern.search(url)
        if match:
            id = match.group(1)
            break

    if not id:
        sys.stderr.write('Unable to get ID from {}'.format(url))
        sys.exit(1)

    process_item(id, directory)

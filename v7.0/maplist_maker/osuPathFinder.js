'use strict';

let regedit = require("regedit");
const readline = require("readline");
const path = require("path");

async function findOsuPath() {
    try {
        let result = await new Promise((res, rej) => {
            regedit.list('HKCR\\osu!\\shell\\open\\command', (e, r) => e ? rej(e) : res(r));
        });
        let osuExePath = result['HKCR\\osu!\\shell\\open\\command'].values[''].value.replace(/^["']/, "").replace(/['"]? ?"%1"/, "");
        const rootPath = osuExePath.replace(/osu!\.exe$/, "");
        return {
            exe: osuExePath,
            root: rootPath,
            songs: path.join(rootPath, "Songs")
        };
    }
    catch(e) {
        const rl = readline.createInterface({
            input: process.stdin,
            output: process.stdout
        });
        return await new Promise(res => {
            rl.question("Input osu! install path (ex: C:\\osu!): ", function(p) {
                p = p.replace(/\//, "\\");
                if(!/\\$/.test(p)) {
                    p += "\\";
                }
                rl.close();
                res({
                    exe: p + "osu!.exe",
                    root: p,
                    songs: path.join(p, "Songs")
                });
            });
        });
    }
}

module.exports = findOsuPath;
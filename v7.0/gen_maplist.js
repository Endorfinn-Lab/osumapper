#!/usr/bin/env node

'use strict';

const osuPathFinder = require("./maplist_maker/osuPathFinder");
const osuDBGetter = require("./maplist_maker/osuDBGetter");
const express = require('express');
const bodyParser = require('body-parser');
const opn = require('opn');
const fs = require('fs');
const path = require('path');

async function main() {
    const osuPaths = await osuPathFinder();

    if (!osuPaths.songs || !fs.existsSync(osuPaths.songs)) {
        console.error(`Could not find osu! Songs folder at "${osuPaths.songs}".`);
        console.error("Please ensure the path is correct. You may be prompted to enter it manually.");
        process.exit(1);
    }

    console.log("Starting to parse beatmaps, this may take a while...");
    const osuDB = osuDBGetter(osuPaths.songs);

    let app = express();
    app.use("/beatmaps", function(req, res, next) {
        res.set("Content-Type", "application/json");
        res.send(JSON.stringify({
            path: osuPaths.root,
            maps: osuDB.beatmaps
        }));
        res.end();
    });
    app.use(bodyParser.json({limit: '200mb'}));
    app.use("/save", function(req, res, next) {
        try {
            let contents = req.body.contents;
            fs.writeFileSync("maplist.txt", contents);
            res.send("saved");
        }
        catch(e) {
            res.send("fail");
        }
        res.end();
    });
    app.use("/", express.static(path.join(__dirname, 'maplist_maker/html')));

    app.use(function(req, res, next) {
        res.status(404);
        res.send("404");
        res.end();
    });

    app.listen(3424, function () {
        const url = "http://127.0.0.1:3424/";
        opn(url);
        console.log(`Node server listening on ${url}`);
        console.log('Parsing complete. Application is ready.');
    });
}
main();

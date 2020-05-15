#![feature(box_patterns)]
#![feature(box_syntax)]

extern crate schemars;
extern crate serde_json;

use schemars::schema::{
    InstanceType, ObjectValidation, RootSchema, Schema, SchemaObject, SingleOrVec,
};

use std::collections::BTreeMap;
use std::fs::File;
use std::io::BufReader;
use std::iter::{self, Zip};
use std::path::{Path, PathBuf};

fn strvec_to_path(components: Vec<String>) -> PathBuf {
    components
        .iter()
        .fold(PathBuf::new(), |acc, c| acc.join(PathBuf::from(c)))
}

fn path_to_components(path: &Path) -> Vec<String> {
    path.components()
        .map(|c| c.as_os_str().to_str().unwrap().to_string())
        .collect()
}

// example:
// within = /a/b/c/d.txt
// to = /a/b/e/f.txt
//
// inside d.txt we reference to via ../e/f.txt
fn relativize(within: &Path, to: &Path) -> PathBuf {
    fn partition_until_diff(path: &Path, comparison: &Path) -> (PathBuf, PathBuf) {
        let mut left = Vec::new();
        let mut right = path_to_components(path);

        for c in comparison.components() {
            let c = c.as_os_str().to_str().unwrap();

            let right_copy = right.clone();
            match right.split_first_mut() {
                Some((first, rest)) => {
                    if first == c {
                        left.push(first.clone());
                        right = rest.to_vec();
                    } else {
                        return (strvec_to_path(left), strvec_to_path(right_copy));
                    }
                }
                None => return (strvec_to_path(left), PathBuf::new()),
            }
        }

        (strvec_to_path(left), strvec_to_path(right))
    }

    let (p2left, p2right) = partition_until_diff(&to, &within);
    let p2s_left: Vec<String> = path_to_components(&p2left);
    let p2s_right: Vec<String> = path_to_components(&p2right);

    let withins: Vec<String> = path_to_components(&within);

    let tos: Vec<String> = path_to_components(&to);

    //println!("p2s_left {:?}, p2s_right {:?}", p2s_left, p2s_right);

    let (_, withins_dir) = withins.split_last().unwrap();
    if p2s_left == withins_dir {
        PathBuf::from("./").join(&p2right)
    } else {
        iter::repeat("../".to_string())
            .take(withins_dir.len() - p2s_left.len())
            .chain(p2s_right.iter().map(|s| s.to_string()))
            .fold(PathBuf::new(), |acc, p| acc.join(p))
    }
}

#[cfg(test)]
mod tests {
    use super::relativize;
    use std::path::PathBuf;

    #[test]
    fn relativize_within_longer() {
        let fixed = relativize(
            &PathBuf::from("a/b/c/d/e.txt"),
            &PathBuf::from("a/x/y/z.txt"),
        );
        assert_eq!(fixed, PathBuf::from("../../../x/y/z.txt"));
    }

    #[test]
    fn relativize_example() {
        let fixed = relativize(
            &PathBuf::from("definitions/automaticRetry/Type"),
            &PathBuf::from("definitions/automaticRetry/properties/exit_status/Type"),
        );
        assert_eq!(fixed, PathBuf::from("./properties/exit_status/Type"));
    }

    #[test]
    fn relativize_same_len_deep_dirs() {
        let fixed = relativize(
            &PathBuf::from("a/b/c/d/e.txt"),
            &PathBuf::from("a/b/x/y/z.txt"),
        );
        assert_eq!(fixed, PathBuf::from("../../x/y/z.txt"));
    }

    #[test]
    fn relativize_same_len_files() {
        let fixed = relativize(&PathBuf::from("a/b/c.txt"), &PathBuf::from("a/b/d.txt"));
        assert_eq!(fixed, PathBuf::from("./d.txt"));
    }

    #[test]
    fn relativize_within_shorter() {
        let fixed = relativize(&PathBuf::from("a/b/c.txt"), &PathBuf::from("a/x/y/z.txt"));
        assert_eq!(fixed, PathBuf::from("../x/y/z.txt"));
    }

    #[test]
    fn relativize_small_path() {
        let fixed = relativize(&PathBuf::from("a/b.txt"), &PathBuf::from("a/x.txt"));
        assert_eq!(fixed, PathBuf::from("./x.txt"));
    }
}

#[derive(Debug, Clone)]
struct VarRef(usize);
impl VarRef {
    fn pp(&self) -> String {
        format!("var{:}", self.0)
    }
}

// Types are stratified (am I using that term right?) because
// this will enforce we always can name any subrecords/unions.
// The nesting unions/records under any other types must be
// indirect via a reference.
//
// TODO: If further symbolic manipulation is necessary consider
// using the {ann, expr} <-> expr mutual recursion trick to add
// an arbitrary annotation to every node.
#[derive(Debug, Clone)]
enum Type0 {
    Var(VarRef),
    Bool,
    Text,
    Natural,
    List(Box<Type0>),
    Optional(Box<Type0>),
    StringMap(Box<Type0>),
    // Relative path, name, arity
    Reference(PathBuf, String, usize),
    Pi(VarRef, Box<Type0>),
}

fn str_cap(s: &str) -> String {
    format!("{}{}", (&s[..1].to_string()).to_uppercase(), &s[1..])
}

impl Type0 {
    fn to_variant_name(&self, ctx: &mut Context) -> String {
        match self {
            Type0::Bool => "Boolean".to_string(),
            Type0::Text => "String".to_string(),
            Type0::Natural => "Number".to_string(),
            Type0::List(box t_) => format!("List{}", t_.to_variant_name(ctx)),
            Type0::Optional(box t_) => format!("Optional{}", t_.to_variant_name(ctx)),
            Type0::StringMap(box t_) => format!("StringMap{}", t_.to_variant_name(ctx)),
            Type0::Reference(_, s, _) => {
                let p = PathBuf::from(s);
                let mut p_ = p.clone();
                // pop to remove detritus
                while path_to_file(&p_).starts_with("Type")
                    || path_to_file(&p_) == "union"
                    || path_to_file(&p_) == "properties"
                {
                    p_.pop();
                }
                let prefix = str_cap(&path_to_file(&p_));
                let rest = path_to_file(&p);
                if prefix == "#" {
                    str_cap(&rest)
                } else {
                    format!("{}/{}", prefix, rest)
                }
            }
            Type0::Pi(var, box t_) => format!("Pi_{}_{}", var.pp(), t_.to_variant_name(ctx)),
            Type0::Var(var) => var.pp(),
        }
    }

    fn pp(&self) -> String {
        match self {
            Type0::Bool => "Bool".to_string(),
            Type0::Text => "Text".to_string(),
            Type0::Natural => "Natural".to_string(),
            Type0::List(box t_) => format!("List {}", t_.pp()),
            Type0::Optional(box t_) => format!("Optional {}", t_.pp()),
            Type0::StringMap(box t_) => format!("List {{ mapKey: Text, mapValue: {} }}", t_.pp()),
            Type0::Reference(path, s, _) => {
                let path = PathBuf::from(&format!(
                    "{}/Type",
                    path.to_str().unwrap().to_string().replace("#/", "")
                ));
                let r = relativize(&path, &PathBuf::from(s.replace("#/", "")))
                    .to_str()
                    .unwrap()
                    .to_string();
                /*println!(
                    "In1: {:?}, In2: {:?}, Out: {:?}",
                    path,
                    s.replace("#/", ""),
                    r
                );*/
                r
            }
            Type0::Pi(var, box t_) => format!("\\({} : Type) -> {}", var.pp(), t_.pp()),
            Type0::Var(var) => var.pp(),
        }
    }
}

#[derive(Debug, Clone)]
enum Type {
    Basic(Type0),
    // union and record can't recurse directly, they must use references
    // this will enforce we always can name any subrecords/unions
    Union(BTreeMap<String, Option<Type0>>),
    Record(BTreeMap<String, Type0>),
}
impl Type {
    fn indirection(&self, name: String, ctx: &mut Context) -> Type0 {
        match self {
            Type::Basic(t0) => t0.clone(),
            u @ Type::Union(_) => {
                ctx.insert(&name, u.clone());
                //ctx_.push("union");
                Type0::Reference(ctx.top_level_path.clone(), ctx.named(&name), 0)
            }
            r @ Type::Record(_) => {
                ctx.insert(&name, r.clone());
                //ctx_.push("properties");
                Type0::Reference(ctx.top_level_path.clone(), ctx.named(&name), 0)
            }
        }
    }

    fn pp(&self) -> String {
        match self {
            Type::Basic(t0) => t0.pp(),
            Type::Union(map) => {
                let mut chunks = vec!["<".to_string()];
                let mut first = true;
                for (key, val) in map.iter() {
                    if !first {
                        chunks.push("|".to_string());
                    }
                    chunks.push(key.to_string());
                    match val {
                        Some(typ) => {
                            chunks.push(":".to_string());
                            chunks.push(typ.pp());
                        }
                        None => (),
                    };

                    first = false;
                }
                chunks.push(">".to_string());
                chunks.join(" ")
            }
            Type::Record(map) => {
                let mut chunks = vec!["{".to_string()];
                let mut first = true;
                for (key, val) in map.iter() {
                    if !first {
                        chunks.push("\n,".to_string());
                    }
                    chunks.push(key.to_string());
                    chunks.push(":".to_string());
                    chunks.push(val.pp());
                    first = false;
                }
                chunks.push("\n}".to_string());
                chunks.join(" ")
            }
        }
    }
}

fn path_to_file(p: &PathBuf) -> String {
    p.file_name().unwrap().to_str().unwrap().to_string()
}

#[derive(Debug, Clone)]
struct Context {
    fresh_: BTreeMap<String, u32>,
    vars: usize,
    path: PathBuf,
    top_level_path: PathBuf,
    data: BTreeMap<String, Type>,
}
impl Context {
    fn new() -> Context {
        Context {
            fresh_: BTreeMap::new(),
            vars: 0,
            path: PathBuf::new().join("#"),
            top_level_path: PathBuf::new().join("/"),
            data: BTreeMap::new(),
        }
    }

    fn new_var(&mut self) -> VarRef {
        let x = self.vars;
        self.vars += 1;
        VarRef(x)
    }

    fn push(&mut self, s: &str) {
        self.path = self.path.join(Path::new(s))
    }

    fn update_top_level_path(&mut self) {
        self.top_level_path = self.path.clone();
    }

    fn pop(&mut self) -> bool {
        self.path.pop()
    }

    fn named(&self, k: &str) -> String {
        let path_ = self.path.join(Path::new(k));
        path_.to_str().unwrap().to_string()
    }

    fn insert(&mut self, k: &str, v: Type) {
        let name = self.named(k);
        self.data.insert(name, v);
    }

    fn fresh(&mut self, name: &str) -> String {
        let key = &self.named(name);
        let result = self.fresh_.get(key).map(|v| *v);
        match result {
            Some(x) => {
                self.fresh_.insert(key.to_string(), x + 1);
                format!("Type{}", x)
            }
            None => {
                self.fresh_.insert(key.to_string(), 0);
                "Type".to_string()
            }
        }
    }

    fn file(&self) -> String {
        path_to_file(&self.path)
    }
}

fn unionOrRegular(ctx: &mut Context, ss: &[Schema]) -> Type {
    match ss {
        [Schema::Object(s)] => schemaToTypeTopLevel(ctx, s),
        xs => {
            let mut map: BTreeMap<String, Option<Type0>> = BTreeMap::new();
            for x in xs.iter() {
                match x {
                    Schema::Object(s) => {
                        ctx.push("union");
                        let typ = schemaToType(ctx, s);
                        ctx.pop();
                        let _ = map.insert(typ.to_variant_name(ctx), Some(typ));
                    }
                    _ => panic!("Unimplemented"),
                }
            }
            Type::Union(map)
        }
    }
}

fn objToType(ctx: &mut Context, o: &ObjectValidation) -> Type {
    // Unsupported
    assert!(o.required.len() == 0);

    let mut map: BTreeMap<String, Type0> = BTreeMap::new();
    ctx.push("properties");
    for (key, val) in o.properties.iter() {
        let obj = schemaAsObj(val);
        ctx.push(key);
        let innerTyp = schemaToType(ctx, obj);
        ctx.pop();
        map.insert(key.clone(), innerTyp);
    }
    ctx.pop();
    Type::Record(map)
}

fn schemaToTypeTopLevel(ctx: &mut Context, s: &SchemaObject) -> Type {
    let s = s.clone();
    match &s.reference {
        Some(_) => Type::Basic(schemaToType(ctx, &s)),
        None => match &s.instance_type {
            Some(sv) => match sv {
                SingleOrVec::Single(it) => match it {
                    box InstanceType::Object => match s.object {
                        Some(o) => objToType(ctx, &o),
                        None => Type::Basic(schemaToType(ctx, &s)),
                    },
                    _ => Type::Basic(schemaToType(ctx, &s)),
                },
                SingleOrVec::Vec(_) => panic!("Unimplemented"),
            },
            None => {
                // if there is no instance type, we can assume "anyOf"
                match s.subschemas {
                    Some(s) => match s.any_of {
                        None => panic!("Unexpected none for any_of"),
                        Some(schemas) => {
                            let s = schemaAsObj(&schemas[0]);
                            if s.instance_type == Some(SingleOrVec::Single(box InstanceType::Null))
                            {
                                Type::Basic(schemaToType(ctx, s))
                            } else {
                                unionOrRegular(ctx, schemas.as_slice())
                            }
                        }
                    },
                    None => objToType(ctx, &s.object.expect("We have an object")),
                }
            }
        },
    }
}

fn schemaToType(ctx: &mut Context, s: &SchemaObject) -> Type0 {
    let s = s.clone();
    match s.reference {
        Some(r) => {
            // HACK: commonOptions needs to be stripped from schema for schemars
            // library to properly parse BuildKite's json schema.
            Type0::Reference(
                ctx.path.clone(),
                format!("{}/Type", r.replace("/commonOptions", "")),
                0,
            )
        }
        None => match s.instance_type {
            Some(sv) => match sv {
                SingleOrVec::Single(it) => match it {
                    box InstanceType::Null => panic!("Unimplemented"),
                    box InstanceType::Boolean => Type0::Bool,
                    box InstanceType::Number | box InstanceType::Integer => Type0::Natural,
                    box InstanceType::String => Type0::Text,
                    box InstanceType::Array => {
                        let array = s.array.expect("Array has array field");
                        let items = array.items.expect("Array has items field");

                        let innerTyp = match items {
                            SingleOrVec::Single(schema) => schemaToType(ctx, schemaAsObj(&schema)),
                            SingleOrVec::Vec(_) => panic!("Unimplemented"),
                        };
                        Type0::List(box innerTyp)
                    }
                    box InstanceType::Object => match s.object {
                        Some(o) => {
                            let name = ctx.file();
                            objToType(ctx, &o).indirection(ctx.fresh(&name), ctx)
                        }
                        None => {
                            let v = ctx.new_var();
                            Type0::Pi(v.clone(), box Type0::StringMap(box Type0::Var(v)))
                        }
                    },
                },
                SingleOrVec::Vec(_) => panic!("Unimplemented"),
            },
            None => {
                // if there is no instance type, we can assume "anyOf"
                match s.subschemas {
                    Some(s) => match s.any_of {
                        None => panic!("Unexpected none for any_of"),
                        Some(schemas) => {
                            let s = schemaAsObj(&schemas[0]);
                            if s.instance_type == Some(SingleOrVec::Single(box InstanceType::Null))
                            {
                                let (_, rest) = schemas
                                    .as_slice()
                                    .split_first()
                                    .expect("We already indexed at 0");
                                let inner = unionOrRegular(ctx, rest);
                                let resolved =
                                    inner.indirection(format!("{}NonNull", ctx.file()), ctx);

                                Type0::Optional(box resolved)
                            } else {
                                let name = ctx.file();
                                unionOrRegular(ctx, schemas.as_slice())
                                    .indirection(ctx.fresh(&name), ctx)
                            }
                        }
                    },
                    None => {
                        let name = ctx.file();
                        objToType(ctx, &s.object.expect("We have an object"))
                            .indirection(ctx.fresh(&name), ctx)
                    }
                }
            }
        },
    }
}

fn schemaAsObj(s: &Schema) -> &SchemaObject {
    match s {
        Schema::Bool(_) => panic!("Unsupported boolean schema"),
        Schema::Object(s) => s,
    }
}

fn addDefinitionsToCtx(ctx: &mut Context, definitions: &BTreeMap<String, Schema>) {
    ctx.push("definitions");
    for (key, value) in definitions.iter() {
        println!("Got key {}", key);
        ctx.push(key);
        ctx.update_top_level_path();
        let innerTyp = schemaToTypeTopLevel(ctx, schemaAsObj(value));
        ctx.pop();

        //match innerTyp {
        //    Type::Record(_) => println!("Skipping record"),
        //    t => {
        ctx.insert(&format!("{}/Type", key), innerTyp);
        //    }
        //};
    }
    ctx.pop();
}

fn rootSchema(r: &RootSchema) -> (Context, Type) {
    let mut ctx = Context::new();
    addDefinitionsToCtx(&mut ctx, &r.definitions);

    let rootRecord = schemaToTypeTopLevel(&mut ctx, &r.schema);

    // println!("Root record {:#?}, ctx {:#?}", rootRecord, ctx);
    (ctx, rootRecord)
}

fn lambda_lift(ctx: &mut Context) {
    // TODO
}

fn main() {
    let file = File::open(PathBuf::new().join("schema.json")).expect("schema");
    let reader = BufReader::new(file);

    let schema: RootSchema = serde_json::from_reader(reader).expect("from_reader");

    let (ctx, rootRecord) = rootSchema(&schema);

    for (key, val) in &ctx.data {
        println!("{:}: {:}", key, val.pp());
    }

    // TODO: Lift the Pi's through references and out of records and unions
    // TODO: Write it out to disk

    // println!("{:#?}", schema);
}
